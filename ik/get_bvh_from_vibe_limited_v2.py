import numpy as np
import pydart2 as pydart
from PyCommon.modules.GUI import DartViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Math import mmMath as mm
from fltk import Fl
import math
from scipy.optimize import minimize, Bounds
import joblib
import subprocess

from SkateUtils.DartMotionEdit import skelqs2bvh, DartSkelMotion

def get_spin_joint_names():
    # vibe 3d joint indices
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
        'rankle',         # 25
        'rknee',          # 26
        'rhip',           # 27
        'lhip',           # 28
        'lknee',          # 29
        'lankle',         # 30
        'rwrist',         # 31
        'relbow',         # 32
        'rshoulder',      # 33
        'lshoulder',      # 34
        'lelbow',         # 35
        'lwrist',         # 36
        'neck',           # 37
        'headtop',        # 38
        'hip',            # 39 'Pelvis (MPII)', # 39
        'thorax',         # 40 'Thorax (MPII)', # 40
        'Spine (H36M)',   # 41
        'Jaw (H36M)',     # 42
        'Head (H36M)',    # 43
        'nose',           # 44
        'leye',           # 45 'Left Eye', # 45
        'reye',           # 46 'Right Eye', # 46
        'lear',           # 47 'Left Ear', # 47
        'rear',           # 48 'Right Ear', # 48
    ]


def get_staf_joint_names():
    # vibe joints2d output
    return [
        'OP Nose', # 0,
        'OP Neck', # 1,
        'OP RShoulder', # 2,
        'OP RElbow', # 3,
        'OP RWrist', # 4,
        'OP LShoulder', # 5,
        'OP LElbow', # 6,
        'OP LWrist', # 7,
        'OP MidHip', # 8,
        'OP RHip', # 9,
        'OP RKnee', # 10,
        'OP RAnkle', # 11,
        'OP LHip', # 12,
        'OP LKnee', # 13,
        'OP LAnkle', # 14,
        'OP REye', # 15,
        'OP LEye', # 16,
        'OP REar', # 17,
        'OP LEar', # 18,
        'Neck (LSP)', # 19,
        'Top of Head (LSP)', # 20,
    ]

name_idx_map = {
    'j_root': 39,
    # 'j_root_2':8

    'j_thigh_left': 28,
    'j_shin_left': 13, 
    # 'j_shin_left_2': 29,
    'j_heel_left': 14,
    # 'j_heel_left_2': 30,
    'j_bigtoe_left': 19,
    'j_smalltoe_left': 20,
    'j_trueheel_left': 21,
    
    'j_thigh_right': 27,
    'j_shin_right': 10,
    # 'j_shin_right_2': 26,
    'j_heel_right': 11,
    # 'j_heel_right_2': 25,
    'j_bigtoe_right': 22,
    'j_smalltoe_right': 23,
    'j_trueheel_right': 24,
    
    # 'j_spine': -1,
    
    'j_neck': 1,
    # 'j_head': -1,
    
    'j_bicep_left': 5,
    # 'j_bicep_left_2': 34,
    'j_forearm_left': 6,
    # 'j_forearm_left_2': 35,
    'j_hand_left': 7,
    # 'j_hand_left_2': 36,

    'j_bicep_right': 2,
    # 'j_bicep_right_2': 33,
    'j_forearm_right': 3,
    # 'j_forearm_right_2': 32,
    'j_hand_right': 4,
    # 'j_hand_right_2': 31,
    }


if __name__ == "__main__":
    video_name = input("video name?")
    # video_name = 'high_jump4'

    frame_rate = subprocess.check_output(f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=s=x:p=0 /home/trif/works/foot_contact_learning/data/{video_name}.mp4", shell=True)
    frame_rate = list(map(float, frame_rate.decode().split('/')))
    fps = int(frame_rate[0] / frame_rate[1] + 0.1)
    print(fps)

    file_path = 'data/vibe/' + video_name +'/vibe_output.pkl'
    data = joblib.load(file_path)
    print(len(data))
    print(data.keys())
    if video_name == 'high_jump4':
        frame_ids = list(map(int, data[2]['frame_ids'])) + list(map(int, data[78]['frame_ids']))
        frame_ids[frame_ids.index(98)] = 105
        frame_ids[frame_ids.index(99)] = 113
    else:
        frame_ids = list(map(int, data[1]['frame_ids']))

    frame_num = int(np.max(frame_ids))+1
    missing_ranges = []
    for _i in range(len(frame_ids)-1):
        if frame_ids[_i+1] - frame_ids[_i] > 1:
            missing_ranges.append((frame_ids[_i], frame_ids[_i+1]))

    if video_name == 'high_jump4':
        joint_3d_infos = np.append(data[2]['joints3d'], data[78]['joints3d'], axis=0)
    else:
        joint_3d_infos = data[1]['joints3d']

    # frame_num = len(joint_3d_infos)
    joint_num = len(joint_3d_infos[0])

    print("frame_num:", frame_num)
    print("joint_num:", joint_num)
    print(joint_3d_infos.shape)

    pydart.init()
    world = pydart.World(1./30., 'data/skel/human_mass_limited_dof_v2.skel')
    skel = world.skeletons[1]

    offset = [skel.body(0).to_world()]
    offset = [np.zeros(3)]
    
    motion = DartSkelMotion()
    motion.fps = fps

    hmr_pos = [None]
    hmr_skel_left_leg = [None]
    hmr_skel_right_leg = [None]
    hmr_skel_left_arm = [None]
    hmr_skel_right_arm = [None]

    viewer_w, viewer_h = 1280, 720
    viewer = hsv.DartViewer(rect=(0, 0, viewer_w + 300, 1 + viewer_h + 55))
    viewer.setMaxFrame(frame_num-1)
    viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255,255,255), yr.POLYGON_FILL))
    viewer.doc.addRenderer('hmr_pos', yr.PointsRenderer(hmr_pos))
    viewer.doc.addRenderer('hmr_skel_left_leg', yr.LinesRenderer(hmr_skel_left_leg, color=(0, 0, 255)))
    viewer.doc.addRenderer('hmr_skel_right_leg', yr.LinesRenderer(hmr_skel_right_leg))
    viewer.doc.addRenderer('hmr_skel_left_arm', yr.LinesRenderer(hmr_skel_left_arm, color=(0, 0, 255)))
    viewer.doc.addRenderer('hmr_skel_right_arm', yr.LinesRenderer(hmr_skel_right_arm))

    viewer.objectInfoWnd.add1DSlider('posX', -1., 1., 0.001, offset[0][0])
    viewer.objectInfoWnd.add1DSlider('posY', 0., 2., 0.001, offset[0][1])
    viewer.objectInfoWnd.add1DSlider('posZ', -1., 1., 0.001, offset[0][2])

    viewer.objectInfoWnd.add1DSlider('rotX', -math.pi, math.pi, 0.001, math.pi)
    viewer.objectInfoWnd.add1DSlider('rotY', -math.pi, math.pi, 0.001, 0.)
    viewer.objectInfoWnd.add1DSlider('rotZ', -math.pi, math.pi, 0.001, 0.)

    rot_offset = [mm.exp([viewer.objectInfoWnd.getVal('rotX'), viewer.objectInfoWnd.getVal('rotY'), viewer.objectInfoWnd.getVal('rotZ')])]

    missing_frames = []

    def simulate_callback(frame):
        del hmr_pos[:]
        del hmr_skel_left_arm[:]
        del hmr_skel_right_arm[:]
        del hmr_skel_left_leg[:]
        del hmr_skel_right_leg[:]

        if frame in frame_ids:
            data_frame = frame_ids.index(frame)
        else:
            missing_frames.append(frame)
            motion.append(skel.q, skel.dq)
            return

        if video_name == "high_jump4":
            def to_rad(deg):
                return deg/180. * math.pi

            if frame == 105:
                q_ = np.zeros_like(skel.q)
                q_[0] = to_rad(-80.)
                q_[skel.dof_index('j_shin_left')] = to_rad(100.)
                q_[skel.dof_index('j_shin_right')] = to_rad(90.)
                q_[skel.dof_index('j_abdomen_x')] = to_rad(-20.)
                q_[skel.dof_index('j_spine_x')] = to_rad(-30.)

                q_[skel.dof_index('j_bicep_right_y')] = to_rad(45.)

                q_[skel.dof_index('j_forearm_right_y')] = to_rad(15.)

                q_[skel.dof_index('j_bicep_left_x')] = to_rad(-35.)
                q_[skel.dof_index('j_bicep_left_y')] = to_rad(45.)
                q_[skel.dof_index('j_bicep_left_z')] = to_rad(-40.)
                skel.set_positions(q_)
                motion.append(skel.q, skel.dq)
                return
            if frame == 113:
                q_ = np.zeros_like(skel.q)
                q_[0] = to_rad(-90.)
                q_[skel.dof_index('j_thigh_left_x')] = to_rad(-95.)
                q_[skel.dof_index('j_thigh_right_x')] = to_rad(-85.)
                q_[skel.dof_index('j_shin_left')] = to_rad(0.)
                q_[skel.dof_index('j_shin_right')] = to_rad(5.)
                q_[skel.dof_index('j_abdomen_x')] = to_rad(15.)
                q_[skel.dof_index('j_spine_x')] = to_rad(30.)

                q_[skel.dof_index('j_bicep_right_y')] = to_rad(45.)
                q_[skel.dof_index('j_bicep_right_z')] = to_rad(25.)

                q_[skel.dof_index('j_forearm_right_y')] = to_rad(25.)

                q_[skel.dof_index('j_bicep_left_x')] = to_rad(-35.)
                q_[skel.dof_index('j_bicep_left_y')] = to_rad(-45.)

                q_[skel.dof_index('j_forearm_left_y')] = to_rad(-25.)
                skel.set_positions(q_)
                motion.append(skel.q, skel.dq)
                return

        offset[0] = skel.body(0).to_world() - joint_3d_infos[data_frame][name_idx_map['j_root']]
        hmr_pos[:] = joint_3d_infos[data_frame]

        for i in range(len(hmr_pos)):
            hmr_pos[i] = np.dot(rot_offset[0], hmr_pos[i] - joint_3d_infos[data_frame][name_idx_map['j_root']]) + skel.body(0).to_world()
        
        hmr_skel_left_leg.append(hmr_pos[name_idx_map['j_thigh_left']])
        hmr_skel_left_leg.append(hmr_pos[name_idx_map['j_shin_left']])
        hmr_skel_left_leg.append(hmr_pos[name_idx_map['j_heel_left']])

        hmr_skel_right_leg.append(hmr_pos[name_idx_map['j_thigh_right']])
        hmr_skel_right_leg.append(hmr_pos[name_idx_map['j_shin_right']])
        hmr_skel_right_leg.append(hmr_pos[name_idx_map['j_heel_right']])

        hmr_skel_left_arm.append(hmr_pos[name_idx_map['j_neck']])
        hmr_skel_left_arm.append(hmr_pos[name_idx_map['j_bicep_left']])
        hmr_skel_left_arm.append(hmr_pos[name_idx_map['j_forearm_left']])
        hmr_skel_left_arm.append(hmr_pos[name_idx_map['j_hand_left']])

        hmr_skel_right_arm.append(hmr_pos[name_idx_map['j_neck']])
        hmr_skel_right_arm.append(hmr_pos[name_idx_map['j_bicep_right']])
        hmr_skel_right_arm.append(hmr_pos[name_idx_map['j_forearm_right']])
        hmr_skel_right_arm.append(hmr_pos[name_idx_map['j_hand_right']])

        joints_weights = {
            'j_thigh_left': 2., 
            'j_shin_left': 1., 
            'j_heel_left': 1., 
            'j_thigh_right': 2., 
            'j_shin_right': 1., 
            'j_heel_right': 1., 
            'j_neck': 1.,
            'j_bicep_right': 1., 
            'j_forearm_right': 1., 
            'j_hand_right': 1.,
            'j_bicep_left': 1., 
            'j_forearm_left': 1., 
            'j_hand_left': 1.
        }

        joints_fix_weights = {
            'j_thigh_left_y': 1.,
            # 'j_shin_left_y': 1.,
            'j_thigh_right_y': 1.,
            # 'j_shin_right_y': 1.,
            'j_abdomen_y': 1.,
            'j_spine_y': 1.,
            'j_scapula_left_x': 1.,
            'j_bicep_left_x': 1.,
            'j_forearm_left_x': 1.,
            'j_hand_left_x': 1.,
            'j_scapula_right_x': 1.,
            'j_bicep_right_x': 1.,
            'j_forearm_right_x': 1.,
            'j_hand_right_x': 1.,
        }

        def ik_f(x):
            q = skel.positions()
            q_ori = q.copy()
            q[:3] = x[:3]
            q[6:] = x[3:]
            skel.set_positions(q)
            sums = 0
            
            for joint_name, joint_weight in joints_weights.items():
                sums += joint_weight * np.linalg.norm(skel.joint(joint_name).position_in_world_frame() - hmr_pos[name_idx_map[joint_name]]) ** 2
            
            for dof_name, joint_fix_weight in joints_fix_weights.items():
                sums += joint_fix_weight * (x[skel.dof_index(dof_name) - 3] ** 2)

            sums += 0.000001 * (np.linalg.norm(x) ** 2)
            skel.set_positions(q_ori)

            return sums
        

        q_0 = np.zeros(skel.ndofs-3)
        q_ = skel.positions()
        q_lb = np.zeros_like(q_0)
        for i in range(len(q_lb)):
            q_lb[i] = -np.inf
        q_lb[skel.joint('j_shin_left').dofs[0].index_in_skeleton()-3] = skel.joint('j_shin_left').dofs[0].position_lower_limit()
        q_lb[skel.joint('j_shin_right').dofs[0].index_in_skeleton()-3] = skel.joint('j_shin_right').dofs[0].position_lower_limit()
        q_ub = np.zeros_like(q_0)
        for i in range(len(q_ub)):
            q_ub[i] = np.inf
        ik_res = minimize(ik_f, q_0, bounds=Bounds(q_lb, q_ub))
        q_answer = ik_res.x
        
        q_[:3] = q_answer[:3]
        q_[6:] = q_answer[3:]
        skel.set_positions(q_)

        def footik_f(x):
            q = skel.positions()
            q_ori = q.copy()
            q[10:13] = x[:3]
            q[17:20] = x[3:]
            skel.set_positions(q)
            sums = 0
            
            sums += np.linalg.norm(skel.body('h_heel_right').to_world([0.0378, -0.0249, -0.10665]) - hmr_pos[name_idx_map['j_bigtoe_right']]) ** 2
            sums += np.linalg.norm(skel.body('h_heel_right').to_world([-0.0378, -0.0249, -0.10665]) - hmr_pos[name_idx_map['j_smalltoe_right']]) ** 2
            sums += np.linalg.norm(skel.body('h_heel_right').to_world([0., -0.0249, 0.10665]) - hmr_pos[name_idx_map['j_trueheel_right']]) ** 2
            sums += np.linalg.norm(skel.body('h_heel_left').to_world([-0.0378, 0.0249, 0.10665]) - hmr_pos[name_idx_map['j_bigtoe_left']]) ** 2
            sums += np.linalg.norm(skel.body('h_heel_left').to_world([0.0378, 0.0249, 0.10665]) - hmr_pos[name_idx_map['j_smalltoe_left']]) ** 2
            sums += np.linalg.norm(skel.body('h_heel_left').to_world([0., 0.0249, -0.10665]) - hmr_pos[name_idx_map['j_trueheel_left']]) ** 2
            
            sums += 0.000001 * (np.linalg.norm(x) ** 2)
            skel.set_positions(q_ori)

            return sums

        q_foot_0 = np.zeros(6)
        q_foot_0[:3] = q_[10:13]
        q_foot_0[3:] = q_[17:20]
        q_foot_answer = minimize(footik_f, q_foot_0).x
        q_[10:13] = q_foot_answer[:3]
        q_[17:20] = q_foot_answer[3:]
        skel.set_positions(q_)
        motion.append(skel.q, skel.dq)
    
    def slider_callback(data):
        offset[0] = np.array([viewer.objectInfoWnd.getVal('posX'), viewer.objectInfoWnd.getVal('posY'), viewer.objectInfoWnd.getVal('posZ')])
        rot_offset[0] = mm.exp([viewer.objectInfoWnd.getVal('rotX'), viewer.objectInfoWnd.getVal('rotY'), viewer.objectInfoWnd.getVal('rotZ')])

    viewer.objectInfoWnd.getValobject('posX').callback(slider_callback)
    viewer.objectInfoWnd.getValobject('posY').callback(slider_callback)
    viewer.objectInfoWnd.getValobject('posZ').callback(slider_callback)

    viewer.objectInfoWnd.getValobject('rotX').callback(slider_callback)
    viewer.objectInfoWnd.getValobject('rotY').callback(slider_callback)
    viewer.objectInfoWnd.getValobject('rotZ').callback(slider_callback)
        
    viewer.setSimulateCallback(simulate_callback)
    viewer.startTimer(1./motion.fps)
    viewer.show()

    Fl.run()

    print(missing_ranges)
    for missing_range in missing_ranges:
        range_length = missing_range[1] - missing_range[0]
        for missing_frame in range(missing_range[0]+1, missing_range[1]):
            range_index = missing_frame - missing_range[0]
            motion.set_q(missing_frame, DartSkelMotion.slerp_general(motion.get_q(missing_range[0]), motion.get_q(missing_range[1]), range_index/range_length, skel), skel.dq)

    motion.refine_dqs(skel)
    motion.save('data/vibe/'+video_name+'/'+video_name+'_vibe_dof_limit_v2.skmo')
