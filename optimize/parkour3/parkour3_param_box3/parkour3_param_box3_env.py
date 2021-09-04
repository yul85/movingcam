import pydart2 as pydart
from SkateUtils.DartMotionEdit import DartSkelMotion
import numpy as np
from math import exp, pi, log, acos, sqrt
from PyCommon.modules.Math import mmMath as mm
from random import random, randrange, uniform
import gym
import gym.spaces
from gym.utils import seeding
import glob
import json
import copy

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
        kp = np.asarray(people['pose_keypoints_2d']).reshape((-1, 3))
        kps.append(kp)
    return kps


class SkateDartEnv(gym.Env):
    def __init__(self, ):
        cur_path = '/'.join(__file__.split('/')[:-1])
        self.world = pydart.World(1. / 300., cur_path + '/../../../data/skel/human_mass_limited_dof_v2.skel')
        self.world.control_skel = self.world.skeletons[1]
        self.skel = self.world.skeletons[1]
        self.Kp, self.Kd = 600., 49.

        self.ref_world = pydart.World(1. / 300., cur_path + '/../../../data/skel/human_mass_limited_dof_v2.skel')
        self.ref_skel = self.ref_world.skeletons[1]
        self.ref_motion = DartSkelMotion()
        self.ref_motion.load(cur_path + '/parkour3.skmo')
        self.ref_motion.reset_root_trajectory_parkour3_optimize(self.ref_skel)
        self.ref_motion.refine_dqs(self.ref_skel)
        self.step_per_frame = 10

        # set self collision
        self.skel.set_self_collision_check(True)
        self.skel.set_adjacent_body_check(False)
        self.skel.body('h_neck').set_collidable(False)
        self.skel.body('h_scapula_left').set_collidable(False)
        self.skel.body('h_scapula_right').set_collidable(False)

        self.box_points = []

        self.height_hat_list = [0.15, 0.15, 0.2, 0.8]
        self.box_height_mean = copy.deepcopy(self.height_hat_list)

        self.box_offset = 0.0

        # draw box
        # add box([name], [size], [color])
        self.box0_size = [2.0, self.height_hat_list[0], 0.4]
        self.box1_size = [2.0, self.height_hat_list[1], 0.2]
        self.box2_size = [0.7, self.height_hat_list[2], 0.3]
        self.box3_size = [3.0, self.height_hat_list[3], 0.3]
        self.box_size = [self.box0_size, self.box1_size, self.box2_size, self.box3_size]

        self.world.skeletons[0].add_box("box0", self.box0_size, [0.6, 0.6, 0.6])
        self.world.skeletons[0].add_box("box1", self.box1_size, [0.75, 0.75, 0.75])
        self.world.skeletons[0].add_box("box2", self.box2_size, [0.85, 0.85, 0.85])
        self.world.skeletons[0].add_box("box3", self.box3_size, [0.95, 0.95, 0.95])

        self.box_pos = []

        self.box_pos.append(np.array([0., 0.5 * self.height_hat_list[0], .0]))
        self.box_pos.append(np.array([0., 0.5 * self.height_hat_list[1], 1.5]))
        self.box_pos.append(np.array([0., 0.5 * self.height_hat_list[2], 3.5]))
        self.box_pos.append(np.array([0., 0.5 * self.height_hat_list[3], 5.5]))
        self.init_target_box_pos = copy.deepcopy(self.box_pos)

        self.pos_reward = 0.
        self.contact_reward = 0.
        self.up_reward = 0.

        self.rsi = True

        self.w_p = 0.35
        self.w_v = 0.1
        self.w_fc = 0.25
        self.w_up = 0.2
        self.w_torque = 0.1
        self.w_root_ori = 0.2

        self.w_par = 0.3
        self.w_exp_h = 0.1
        self.w_h = 0.5

        self.exp_p = 2. * 6.
        self.exp_v = 0.1 * 6.
        self.exp_fc = 1. * 2.
        self.exp_up = 5.
        self.exp_torque = 1.
        self.exp_root_ori = 1.

        self.exp_par = 5.

        self.exp_exp_h = 5.
        self.exp_h = 15.

        self.body_num = self.skel.num_bodynodes()
        self.reward_bodies = [body for body in self.skel.bodynodes]
        self.reward_boxes = [body for body in self.world.skeletons[0].bodynodes[1:]]

        self.motion_len = len(self.ref_motion)
        self.motion_time = len(self.ref_motion) / self.ref_motion.fps

        self.current_frame = 0
        self.count_frame = 0
        self.max_frame = 30 * 10

        state_num = len(self.state())
        action_num = self.skel.num_dofs() - 6

        state_high = np.array([np.finfo(np.float32).max] * state_num, dtype=np.float32)
        action_high = np.array([pi * 10. / 2.] * action_num, dtype=np.float32)

        self.action_space = gym.spaces.Box(-action_high, action_high)
        self.observation_space = gym.spaces.Box(-state_high, state_high)

        self.viewer = None

        self.ext_force = np.zeros(3)
        self.ext_force_duration = 0.

        self.p_fc = None
        self.p_fc_hat = None
        self.foot_contact_violation = False
        self.contact_violation_num = 0
        self.is_foot_contact_same_list = []

        self.multi_body_collision = False
        self.contact_info = []

        self.up_angle_diff = 0.
        self.up_angle_list = []

        self.test = False
        self.test_time = 8

        # box0
        self.lf_contact_start_frame1 = 0
        self.lf_contact_end_frame1 = 2

        # box1
        self.rf_contact_start_frame1 = 6
        self.rf_contact_end_frame1 = 12

        # box2
        self.lf_contact_start_frame2 = 21
        self.lf_contact_end_frame2 = 27

        # box3
        self.both_f_contact_start_frame1 = 40
        self.both_f_contact_end_frame1 = 75

        self.step_torques = []

        file_name = "parkour3"
        json_path = '../../../data/openpose/' + file_name + '/'

        prev_keypoint_backup = None
        for json_file in sorted(glob.glob(json_path + "*.json")):
            keypoint = read_json(json_file)

            if len(keypoint) == 0:
                # print("NO OPENPOSE RESULT!!", json_file)
                keypoint = prev_keypoint_backup

            head_pos = np.array([keypoint[0][1][0], keypoint[0][1][1]])
            midhip_pos = np.array([keypoint[0][8][0], keypoint[0][8][1]])

            openpose_up_vec = midhip_pos - head_pos
            y_vec = np.array([0, 1])

            up_angle = acos(np.dot(openpose_up_vec, y_vec) / np.linalg.norm(openpose_up_vec))
            self.up_angle_list.append(up_angle)

            prev_keypoint_backup = keypoint

            if json_file == json_path + "parkour3_000000000065_keypoints.json":
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

        box_p = np.array(
            [(body.to_world([0, 0.5 * self.box_height_mean[box_idx], 0]) - p_pelvis) / 10. for box_idx, body
             in
             enumerate(self.reward_boxes)]).flatten()

        state.extend(box_p)

        return np.asarray(state).flatten()

    def reward(self):
        self.ref_skel.set_positions(self.ref_motion.get_q(self.current_frame))
        self.ref_skel.set_velocities(self.ref_motion.get_dq(self.current_frame))

        r_p = exp_reward_term(self.w_p, self.exp_p,
                              self.skel.position_differences(self.skel.q, self.ref_skel.q)[6:] / sqrt(
                                  len(self.skel.q[6:])))
        r_v = exp_reward_term(self.w_v, self.exp_v,
                              self.skel.velocity_differences(self.skel.dq, self.ref_skel.dq)[6:] / sqrt(
                                  len(self.skel.dq[6:])))

        self.pos_reward += r_p

        # foot contact reward
        fc_diff = np.clip(np.abs(self.p_fc - self.p_fc_hat), 0., 1.)
        r_fc = exp_reward_term(self.w_fc, self.exp_fc, fc_diff / sqrt(len(fc_diff)))

        self.contact_reward += r_fc
        # load input up angle data
        # up_angle_hat: given input obtained from video
        up_angle_hat = self.up_angle_list[self.current_frame]

        head_pos = self.skel.joint('j_head').position_in_world_frame()
        pelvis_pos = self.skel.joint('j_pelvis').position_in_world_frame()

        sim_up_vec = head_pos - pelvis_pos
        y_axis = np.array([0, 1., 0])

        up_angle = acos(np.dot(sim_up_vec, y_axis) / np.linalg.norm(sim_up_vec))

        self.up_angle_diff = abs(up_angle - up_angle_hat)
        r_up = exp_reward_term(self.w_up, self.exp_up, [self.up_angle_diff])
        self.up_reward += r_up

        # minimize joint torque
        torque = sum(self.step_torques) / len(self.step_torques)
        r_torque = exp_reward_term(self.w_torque, self.exp_torque, torque / sqrt(len(torque)))

        reward = r_p + r_v + r_fc + r_up + r_torque

        # pelvis orientation
        if (self.rf_contact_start_frame1 - 2 <= self.current_frame <= self.rf_contact_end_frame1) \
                or (self.lf_contact_start_frame2 - 2 <= self.current_frame <= self.lf_contact_end_frame2) \
                or (self.both_f_contact_start_frame1 - 2 <= self.current_frame <= self.both_f_contact_end_frame1):
            r_root_ori = exp_reward_term(self.w_root_ori, self.exp_root_ori,
                                         [1. - np.dot(self.skel.body(0).world_transform()[:3, 0], mm.unitX())])
            reward = (1. - self.w_root_ori) * reward + r_root_ori

        # parabola hint
        parabola_hint_ranges = [
            (self.lf_contact_end_frame1, self.rf_contact_start_frame1, ['h_heel_right'], 1),
            (self.rf_contact_end_frame1, self.lf_contact_start_frame2, ['h_heel_left'], 2),
            (self.lf_contact_end_frame2, self.both_f_contact_start_frame1, ['h_heel_left', 'h_heel_right'], 3),
        ]
        for parabola_hint_range in parabola_hint_ranges:
            parabola_box_idx = parabola_hint_range[3]
            if parabola_hint_range[0] < self.current_frame < parabola_hint_range[1]:
                dt = (parabola_hint_range[1] - self.current_frame) / self.ref_motion.fps
                exp_com_pos = self.skel.com() + dt * self.skel.com_velocity() + 0.5 * dt * dt * self.world.gravity()
                exp_diff_com_to_contact = self.box_pos[parabola_box_idx] + self.box_height_mean[
                    parabola_box_idx] * mm.unitY() / 2. - exp_com_pos
                norm_exp_diff_com_to_contact = np.linalg.norm(exp_diff_com_to_contact)
                self.ref_skel.set_positions(self.ref_motion.get_q(parabola_hint_range[1]))
                pos_diff_ref_com_to_ref_contact = \
                    [norm_exp_diff_com_to_contact - np.linalg.norm(
                        self.ref_skel.body(body_name).to_world() - self.ref_skel.com()) for body_name in
                     parabola_hint_range[2]]
                self.ref_skel.set_positions(self.ref_motion.get_q(self.current_frame))
                self.ref_skel.set_velocities(self.ref_motion.get_dq(self.current_frame))
                r_par = exp_reward_term(self.w_par, self.exp_par, np.asarray(pos_diff_ref_com_to_ref_contact) / sqrt(
                    len(pos_diff_ref_com_to_ref_contact)))
                reward = (1. - self.w_par) * reward + r_par
                break

        # height hint
        exp_contact_body_height_diff = []

        if self.rf_contact_start_frame1 - 2 <= self.current_frame < self.rf_contact_start_frame1:
            left_frame = self.rf_contact_start_frame1 - self.current_frame
            exp_contact_body_height_diff.append(
                self.skel.body('h_heel_right').to_world()[1] - self.height_hat_list[1] - 0.0249 - 0.1 * left_frame)

        if self.lf_contact_start_frame2 - 2 <= self.current_frame < self.lf_contact_start_frame2:
            left_frame = self.lf_contact_start_frame2 - self.current_frame
            exp_contact_body_height_diff.append(
                self.skel.body('h_heel_left').to_world()[1] - self.height_hat_list[2] - 0.0249 - 0.1 * left_frame)

        if self.both_f_contact_start_frame1 - 2 <= self.current_frame < self.both_f_contact_start_frame1:
            left_frame = self.both_f_contact_start_frame1 - self.current_frame
            exp_contact_body_height_diff.append(
                self.skel.body('h_heel_left').to_world()[1] - self.height_hat_list[3] - 0.0249 - 0.1 * left_frame)
            exp_contact_body_height_diff.append(
                self.skel.body('h_heel_right').to_world()[1] - self.height_hat_list[3] - 0.0249 - 0.1 * left_frame)
            exp_contact_body_height_diff.append(
                self.skel.body('h_heel_left').to_world()[1] - self.skel.body('h_heel_right').to_world()[1])

        if len(exp_contact_body_height_diff) > 0:
            r_exp_h = exp_reward_term(self.w_exp_h, self.exp_exp_h, np.asarray(exp_contact_body_height_diff) / sqrt(
                len(exp_contact_body_height_diff)))
            reward = (1. - self.w_exp_h) * reward + r_exp_h

        contact_body_height_diff = []
        if self.lf_contact_start_frame1 <= self.current_frame <= self.lf_contact_end_frame1:
            contact_body_height_diff.append(
                max(0., abs(self.skel.body('h_heel_left').to_world()[0] - self.box_pos[0][0]) - self.box0_size[0] / 4.))
            contact_body_height_diff.append(
                self.skel.body('h_heel_left').to_world()[1] - self.height_hat_list[0] - 0.0249)
            contact_body_height_diff.append(self.skel.body('h_heel_left').to_world()[2] - self.box_pos[0][2])

        if self.rf_contact_start_frame1 <= self.current_frame <= self.rf_contact_end_frame1:
            contact_body_height_diff.append(max(0.,
                                                abs(self.skel.body('h_heel_right').to_world()[0] - self.box_pos[1][0]) -
                                                self.box1_size[0] / 4.))
            contact_body_height_diff.append(
                self.skel.body('h_heel_right').to_world()[1] - self.height_hat_list[1] - 0.0249)
            contact_body_height_diff.append(self.skel.body('h_heel_right').to_world()[2] - self.box_pos[1][2])

        if self.lf_contact_start_frame2 <= self.current_frame <= self.lf_contact_end_frame2:
            contact_body_height_diff.append(
                max(0., abs(self.skel.body('h_heel_left').to_world()[0] - self.box_pos[2][0]) - self.box2_size[0] / 4.))
            contact_body_height_diff.append(
                self.skel.body('h_heel_left').to_world()[1] - self.height_hat_list[2] - 0.0249)
            contact_body_height_diff.append(self.skel.body('h_heel_left').to_world()[2] - self.box_pos[2][2])

        if self.both_f_contact_start_frame1 <= self.current_frame <= self.both_f_contact_end_frame1:
            contact_body_height_diff.append(
                max(0., abs(self.skel.body('h_heel_left').to_world()[0] - self.box_pos[3][0]) - self.box3_size[0] / 4.))
            contact_body_height_diff.append(
                self.skel.body('h_heel_left').to_world()[1] - self.height_hat_list[3] - 0.0249)
            contact_body_height_diff.append(self.skel.body('h_heel_left').to_world()[2] - self.box_pos[3][2])
            contact_body_height_diff.append(max(0.,
                                                abs(self.skel.body('h_heel_right').to_world()[0] - self.box_pos[3][0]) -
                                                self.box3_size[0] / 4.))
            contact_body_height_diff.append(
                self.skel.body('h_heel_right').to_world()[1] - self.height_hat_list[3] - 0.0249)
            contact_body_height_diff.append(self.skel.body('h_heel_right').to_world()[2] - self.box_pos[3][2])
            contact_body_height_diff.append(
                self.skel.body('h_heel_left').to_world()[1] - self.skel.body('h_heel_right').to_world()[1])

        if len(contact_body_height_diff) > 0:
            r_h = exp_reward_term(self.w_h, self.exp_h,
                                  np.asarray(contact_body_height_diff) / sqrt(len(contact_body_height_diff)))
            reward = (1. - self.w_h) * reward + r_h

        return reward

    def is_done(self):
        if self.current_frame == self.both_f_contact_end_frame1:
            if PRINT_MODE:
                print("Window end!!")
            return True
        if self.multi_body_collision:
            if PRINT_MODE:
                print("multiple body collision with a box occurs!")
            return True

        if self.foot_contact_violation:
            if PRINT_MODE:
                print('not follow the contact hint too long', self.current_frame)
            return True

        if self.skel.com()[1] < 0.3:
            if PRINT_MODE:
                print('fallen')
            return True
        if self.up_angle_diff > 30. / 180. * pi:
            if PRINT_MODE:
                print('up angle diff')
            return True

        if self.skel.body('h_head').id in [contact_info[1] for contact_info in self.contact_info]:
            return True

        if True in np.isnan(np.asarray(self.skel.q)) or True in np.isnan(np.asarray(self.skel.dq)):
            # print('nan')
            return True

        if self.ref_motion.has_loop and self.count_frame >= self.max_frame:
            # print('timeout1')
            return True

        if not self.ref_motion.has_loop and self.current_frame == self.motion_len - 1:
            # print('timeout2')
            return True

        return False

    def step(self, _action):
        action = np.hstack((np.zeros(6), _action / 10.))

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

        # ----------------------------------------------
        del self.contact_info[:]
        for contact in self.world.collision_result.contacts:
            if contact.skel_id1 == 0 and contact.skel_id2 == 1:
                self.contact_info.append((contact.bodynode_id1, contact.bodynode_id2))
            elif contact.skel_id1 == 1 and contact.skel_id2 == 0:
                self.contact_info.append((contact.bodynode_id2, contact.bodynode_id1))

        self.contact_info = list(set(self.contact_info))

        if self.contact_info:
            _, skel_only_contact_info = zip(*self.contact_info)
        else:
            skel_only_contact_info = []
        skel_only_contact_info = list(skel_only_contact_info)
        end_effector_contact_num = sum([c_info[1] in [3, 6, 14, 18] for c_info in self.contact_info])
        no_end_effector_contact_num = len(self.contact_info) - end_effector_contact_num
        self.multi_body_collision = no_end_effector_contact_num > 0

        # box contact info
        # -1 : no contact or multiple contact
        # 0: ground
        # 1: box0
        # 2: box1
        # 3: box2
        # 4: box3
        # 5: box4

        self.p_fc = np.array([-1, -1, -1, -1])
        body_fc_indices = [3, 6, 14, 18]
        for fc_idx, body_fc_index in enumerate(body_fc_indices):
            if skel_only_contact_info.count(body_fc_index) == 1:
                self.p_fc[fc_idx] = self.contact_info[skel_only_contact_info.index(body_fc_index)][0]

        self.p_fc_hat = np.array([-1, -1, -1, -1])

        if self.lf_contact_start_frame1 <= self.current_frame <= self.lf_contact_end_frame1:
            self.p_fc_hat[0] = 1

        if self.rf_contact_start_frame1 <= self.current_frame <= self.rf_contact_end_frame1:
            self.p_fc_hat[1] = 2

        if self.lf_contact_start_frame2 <= self.current_frame <= self.lf_contact_end_frame2:
            self.p_fc_hat[0] = 3

        if self.both_f_contact_start_frame1 <= self.current_frame <= self.both_f_contact_end_frame1:
            self.p_fc_hat[0] = 4
            self.p_fc_hat[1] = 4

        diff_num = sum(self.p_fc[fc_idx] != self.p_fc_hat[fc_idx] for fc_idx in range(len(self.p_fc)))
        self.is_foot_contact_same_list.append(1 if diff_num > 0 else 0)

        self.foot_contact_violation = sum(self.is_foot_contact_same_list) > 3

        return tuple([self.state(), self.reward(), self.is_done(), {}])

    def continue_from_frame(self, frame):
        self.current_frame = frame
        self.ref_skel.set_positions(self.ref_motion.get_q(self.current_frame))

    def reset_optimized_box(self, init_q, init_dq, box_pos_list):
        self.world.reset()
        self.ref_motion.reset_root_trajectory_parkour3_optimize(self.ref_skel)
        self.ref_motion.refine_dqs(self.ref_skel)
        self.continue_from_frame(self.lf_contact_start_frame2)
        self.skel.set_positions(init_q)
        self.skel.set_velocities(init_dq)

        self.count_frame = 0
        self.foot_contact_violation = False
        self.contact_violation_num = 0

        del self.contact_info[:]
        del self.is_foot_contact_same_list[:]

        box_q = self.world.skeletons[0].q
        box_q[9:12] = self.box_pos[0]
        box_q[15:18] = box_pos_list[0]
        box_q[21:24] = box_pos_list[1]
        box_q[24:27] = np.array([-0.2, 0., 0.])
        box_q[27:30] = box_pos_list[2]
        self.world.skeletons[0].set_positions(box_q)

        for i in range(1, len(self.height_hat_list)):
            self.height_hat_list[i] = 0.5 * self.box_size[i][1] + self.box_pos[i][1]

        self.pos_reward = 0.
        self.contact_reward = 0.
        self.up_reward = 0.

        return self.state()

    def reset_init_distribution(self, init_q, init_dq, init_box_pos):
        self.world.reset()
        self.ref_motion.reset_root_trajectory_parkour3_optimize(self.ref_skel)
        self.ref_motion.refine_dqs(self.ref_skel)
        self.continue_from_frame(self.lf_contact_start_frame2)
        self.skel.set_positions(init_q)
        self.skel.set_velocities(init_dq)

        self.box_pos[2] = init_box_pos
        self.height_hat_list[2] = init_box_pos[2]+0.5*self.box_height_mean[2]

        init_target_box_pos = np.array(self.init_target_box_pos[3])
        init_target_pos_x = init_target_box_pos[2]
        init_target_pos_y = init_target_box_pos[1]
        distance_from_prev_box = init_target_pos_x - self.box_pos[2][2]
        distance_margin = 0.2 * distance_from_prev_box
        init_target_height = init_target_box_pos[1] * 2.
        h_margin = 0.2 * init_target_height
        z_val = uniform(init_target_pos_x - distance_margin, init_target_pos_x + distance_margin)
        y_val = uniform(init_target_pos_y - h_margin, init_target_pos_y + h_margin)
        target_box_pos = np.array([self.box_pos[3][0], y_val, z_val])

        self.count_frame = 0
        self.foot_contact_violation = False
        self.contact_violation_num = 0

        del self.contact_info[:]
        del self.is_foot_contact_same_list[:]

        if PRINT_MODE:
            print("target pos: ", target_box_pos)

        box_q = self.world.skeletons[0].q
        box_q[9:12] = self.box_pos[0]
        box_q[15:18] = self.box_pos[1]
        box_q[21:24] = self.box_pos[2]
        box_q[24:27] = np.array([-0.2, 0., 0.])
        box_q[27:30] = target_box_pos
        self.target_box_diff = init_target_box_pos - np.array(box_q[27:30])
        self.world.skeletons[0].set_positions(box_q)

        for i in range(3, len(self.height_hat_list)):
            self.height_hat_list[i] = 0.5 * self.box_size[i][1] + self.box_pos[i][1]

        self.pos_reward = 0.
        self.contact_reward = 0.
        self.up_reward = 0.

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
