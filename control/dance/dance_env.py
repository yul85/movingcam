import pydart2 as pydart
from SkateUtils.DartMotionEdit import DartSkelMotion
import numpy as np
from math import exp, pi, log, acos, sqrt
from PyCommon.modules.Math import mmMath as mm
from random import random, randrange
import gym
import gym.spaces
from gym.utils import seeding
import glob
import json


PRINT_MODE = True
PRINT_MODE = False

def exp_reward_term(w, exp_w, v):
    norm_sq = v * v if isinstance(v, float) else sum(_v * _v for _v in v)
    return w * exp(-exp_w * norm_sq)

def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps


class SkateDartEnv(gym.Env):
    def __init__(self, ):
        cur_path = '/'.join(__file__.split('/')[:-1])
        self.world = pydart.World(1./144., cur_path+'/../../data/skel/human_mass_limited_dof.skel')
        self.world.control_skel = self.world.skeletons[1]
        self.skel = self.world.skeletons[1]
        self.Kp, self.Kd = 400., 40.

        self.ref_world = pydart.World(1./144., cur_path+'/../../data/skel/human_mass_limited_dof.skel')
        self.ref_skel = self.ref_world.skeletons[1]
        self.ref_motion = DartSkelMotion()
        self.ref_motion.load(cur_path + '/dance.skmo')
        self.ref_motion.reset_root_trajectory_dance(self.ref_skel)
        self.ref_motion.refine_dqs(self.ref_skel)
        self.step_per_frame = 6

        self.rsi = True

        # set self collision
        self.skel.set_self_collision_check(True)
        self.skel.set_adjacent_body_check(False)
        self.skel.body('h_neck').set_collidable(False)
        self.skel.body('h_scapula_left').set_collidable(False)
        self.skel.body('h_scapula_right').set_collidable(False)

        # set dof limit
        q_max = np.zeros(self.skel.ndofs)
        q_min = np.zeros(self.skel.ndofs)
        for ref_pose_idx in range(len(self.ref_motion)):
            q = self.ref_motion.get_q(ref_pose_idx)
            q_max = np.maximum(q, q_max)
            q_min = np.minimum(q, q_min)

        joint: pydart.Joint
        for joint_idx, joint in enumerate(self.skel.joints):
            if joint.name.split('_')[1] in ['thigh', 'shin', 'heel']:
                dof: pydart.Dof
                for dof_idx, dof in enumerate(joint.dofs):
                    dof.set_position_upper_limit(q_max[dof.index_in_skeleton()])
                    dof.set_position_lower_limit(q_min[dof.index_in_skeleton()])
                joint.set_position_limit_enforced(True)

            if joint.name.split('_')[1] in ['abdomen', 'spine']:
                dof: pydart.Dof
                for dof_idx, dof in enumerate(joint.dofs):
                    dof.set_position_upper_limit(q_max[dof.index_in_skeleton()] + pi / 16)
                    dof.set_position_lower_limit(q_min[dof.index_in_skeleton()] - pi / 16)
                joint.set_position_limit_enforced(True)

        self.w_p = 0.35
        self.w_v = 0.1
        self.w_up = 0.2
        self.w_fc = 0.25
        self.w_torque = 0.1

        self.exp_p = 2. *6.
        self.exp_v = 0.1*6.
        self.exp_fc = 1.*2.
        self.exp_up = 5.
        self.exp_torque = 1.

        self.body_num = self.skel.num_bodynodes() - 2
        self.reward_bodies = [body for body in self.skel.bodynodes]
        self.motion_len = len(self.ref_motion)
        self.motion_time = len(self.ref_motion) / self.ref_motion.fps

        self.current_frame = 0
        self.count_frame = 0
        self.max_frame = 30 * 10

        state_num = len(self.state())
        action_num = self.skel.num_dofs() - 6

        state_high = np.array([np.finfo(np.float32).max] * state_num)
        action_high = np.array([pi*10./2.] * action_num)

        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-state_high, state_high, dtype=np.float32)

        self.viewer = None

        self.ext_force = np.zeros(3)
        self.ext_force_duration = 0.

        self.p_fc = None
        self.p_fc_hat = np.array([0, 0, 0, 0])
        self.is_foot_contact_same = 0

        self.up_angle_list = []

        self.prev_root_rot = self.ref_motion.get_q(0)[0:3]

        self.step_torques = []

        file_name = "dance"

        # foot contact labels : user input -> network output
        self.contact_label = np.load('../../data/contact_estimation/' + file_name + '_contact_info_foot.npy')

        json_path = '../../data/openpose/' + file_name + '/'

        for json_file in sorted(glob.glob(json_path + "*.json")):
            keypoint = read_json(json_file)

            head_pos = np.array([keypoint[0][1][0], keypoint[0][1][1]])
            midhip_pos = np.array([keypoint[0][8][0], keypoint[0][8][1]])

            openpose_up_vec = midhip_pos - head_pos
            y_vec = np.array([0, 1])

            up_angle = acos(np.dot(openpose_up_vec, y_vec) / np.linalg.norm(openpose_up_vec))

            self.up_angle_list.append(up_angle)

            if json_file == json_path+"dance_000000000068_keypoints.json":
                for i in range(10):
                    self.up_angle_list.append(up_angle)

    def state(self):
        pelvis = self.skel.body(0)
        p_pelvis = pelvis.world_transform()[:3, 3]
        R_pelvis = pelvis.world_transform()[:3, :3]

        phase = self.ref_motion.get_frame_looped(self.current_frame) / self.motion_len
        state = [phase]

        p = np.array([np.dot(R_pelvis.T, body.to_world() - p_pelvis) for body in self.skel.bodynodes[1:]]).flatten()
        R = np.array(
            [mm.rot2quat(np.dot(R_pelvis.T, body.world_transform()[:3, :3])) for body in self.skel.bodynodes]).flatten()
        R[:4] = np.asarray(mm.rot2quat(R_pelvis))
        v = np.array([np.dot(R_pelvis.T, body.world_linear_velocity()) for body in self.skel.bodynodes]).flatten()
        w = np.array(
            [np.dot(R_pelvis.T, body.world_angular_velocity()) / 20. for body in self.skel.bodynodes]).flatten()

        state.extend(p)
        state.extend(np.array([p_pelvis[1]]))
        state.extend(R)
        state.extend(v)
        state.extend(w)

        return np.asarray(state).flatten()

    def reward(self):
        self.ref_skel.set_positions(self.ref_motion.get_q(self.current_frame))
        self.ref_skel.set_velocities(self.ref_motion.get_dq(self.current_frame))

        r_p = exp_reward_term(self.w_p, self.exp_p, self.skel.position_differences(self.skel.q, self.ref_skel.q)[6:] / sqrt(
                                  len(self.skel.q[6:])))
        r_v = exp_reward_term(self.w_v, self.exp_v, self.skel.velocity_differences(self.skel.dq, self.ref_skel.dq)[6:] / sqrt(
                                  len(self.skel.dq[6:])))

        p_fc = np.array([0, 0, 0, 0])

        contact_bodies0 = [contact.bodynode_id1 for contact in self.world.collision_result.contacts]
        contact_bodies1 = [contact.bodynode_id2 for contact in self.world.collision_result.contacts]

        if 3 in contact_bodies0 or 3 in contact_bodies1:
            p_fc[0] = 1
            # print('left foot')

        if 6 in contact_bodies0 or 6 in contact_bodies1:
            p_fc[1] = 1
            # print('right foot')

        if 14 in contact_bodies0 or 14 in contact_bodies1:
            p_fc[2] = 1
            # print('left hand')

        if 18 in contact_bodies0 or 18 in contact_bodies1:
            p_fc[3] = 1
            # print('right hand')

        self.p_fc = p_fc
        fc_diff = np.clip(np.abs(self.p_fc - self.p_fc_hat), 0., 1.)
        r_fc = exp_reward_term(self.w_fc, self.exp_fc, fc_diff/sqrt(len(fc_diff)))

        # up_angle_hat: given input obtained from video
        up_angle_hat = self.up_angle_list[self.current_frame]

        head_pos = self.skel.joint('j_head').position_in_world_frame()
        pelvis_pos = self.skel.joint('j_pelvis').position_in_world_frame()

        sim_up_vec = head_pos - pelvis_pos
        y_axis = np.array([0, 1., 0])

        up_angle = acos(np.dot(sim_up_vec, y_axis) / np.linalg.norm(sim_up_vec))
        self.up_angle_diff = abs(up_angle - up_angle_hat)
        r_up = exp_reward_term(self.w_up, self.exp_up, [self.up_angle_diff])

        # minimize joint torque
        torque = sum(self.step_torques)/len(self.step_torques)
        r_torque = exp_reward_term(self.w_torque, self.exp_torque, torque/sqrt(len(torque)))

        reward = r_p + r_v + r_fc + r_up + r_torque

        return reward

    def is_done(self):
        if self.is_foot_contact_same > 5:
            if PRINT_MODE: print('not follow the contact hint too long', self.current_frame)
            return True
        if self.skel.com()[1] < 0.3:
            if PRINT_MODE: print('fallen')
            return True
        if self.up_angle_diff > 60. / 180. * pi:
            if PRINT_MODE:
                print('up angle diff')
            return True
        if self.skel.body('h_head') in self.world.collision_result.contacted_bodies:
            if PRINT_MODE: print('head contact')
            return True
        if True in np.isnan(np.asarray(self.skel.q)) or True in np.isnan(np.asarray(self.skel.dq)):
            if PRINT_MODE: print('nan')
            return True
        if self.ref_motion.has_loop and self.count_frame >= self.max_frame:
            if PRINT_MODE: print('timeout1')
            return True
        if not self.ref_motion.has_loop and self.current_frame == self.motion_len - 1:
            if PRINT_MODE: print('timeout2')
            return True
        return False

    def step(self, _action):
        action = np.hstack((np.zeros(6), _action/10.))

        next_frame = self.current_frame + 1
        self.ref_skel.set_positions(self.ref_motion.get_q(next_frame))
        self.ref_skel.set_velocities(self.ref_motion.get_dq(next_frame))

        h = self.world.time_step()
        q_des = self.ref_skel.q + action

        del self.step_torques[:]
        for i in range(self.step_per_frame):
            if self.ext_force_duration > 0.:
                self.skel.body('h_spine').add_ext_force(self.ext_force, _isForceLocal=True)
                self.ext_force_duration -= h
                if self.ext_force_duration < 0.:
                    self.ext_force_duration = 0.
            torques = self.skel.get_spd_forces(q_des, self.Kp, self.Kd)
            self.step_torques.append(torques)
            self.skel.set_forces(torques)
            self.world.step()

        self.current_frame = next_frame
        self.count_frame += 1

        # contact
        if self.current_frame > 65:
            self.p_fc_hat[0] = 1
            self.p_fc_hat[1] = 1
        else:
            self.p_fc_hat[0] = self.contact_label[self.current_frame][1]
            self.p_fc_hat[1] = self.contact_label[self.current_frame][0]

        if self.current_frame < 3:
            self.p_fc_hat[0] = 1
            self.p_fc_hat[1] = 1

        if self.current_frame > 63:
            self.p_fc_hat[1] = 1

        if self.current_frame ==23 or self.current_frame == 24:
            self.p_fc_hat[0] = 1

        if not np.array_equal(self.p_fc, self.p_fc_hat):
            # print("false in")
            self.is_foot_contact_same += 1

        return tuple([self.state(), self.reward(), self.is_done(), {}])

    def continue_from_frame(self, frame):
        self.current_frame = frame
        self.ref_skel.set_positions(self.ref_motion.get_q(self.current_frame))
        skel_pelvis_offset = self.skel.joint(0).position_in_world_frame() - self.ref_skel.joint(0).position_in_world_frame()
        skel_pelvis_offset[1] = 0.
        self.ref_motion.translate_by_offset(skel_pelvis_offset)

    def reset(self):
        self.world.reset()
        self.ref_motion.reset_root_trajectory_dance(self.ref_skel)
        self.ref_motion.refine_dqs(self.ref_skel)
        self.continue_from_frame(0)
        self.skel.set_positions(self.ref_motion.get_q(self.current_frame))
        self.skel.set_velocities(np.asarray(self.ref_motion.get_dq(self.current_frame)))
        self.count_frame = 0
        self.is_foot_contact_same = 0

        return self.state()

    def reset_with_q_dq(self, q, dq):
        self.world.reset()
        self.skel.set_positions(q)
        self.skel.set_velocities(dq)
        self.continue_from_frame(0)

        return self.state()

    def reset_with_x_vel(self, x_vel=-1.5):
        self.world.reset()
        self.ref_motion.set_avg_x_vel(x_vel)
        self.ref_motion.refine_dqs(self.ref_skel)
        self.continue_from_frame(randrange(self.motion_len-1) if self.rsi else 0)
        self.skel.set_positions(self.ref_motion.qs[self.current_frame])
        self.skel.set_velocities(np.asarray(self.ref_motion.dqs[self.current_frame]))
        self.count_frame = 0

        return self.state()

    def render(self, mode='human', close=False):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def flag_rsi(self, rsi=True):
        self.rsi = rsi

    def hard_reset(self):
        self.__init__()
        self.reset()
