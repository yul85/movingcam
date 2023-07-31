import math
import numpy as np
import pydart2 as pydart
from scipy.spatial.transform.rotation import Rotation
from PyCommon.modules.Math import mmMath as mm

dart2bvh_joint_index_map = [
    0,  # j_pelvis
    54,  # j_thigh_left
    57,  # j_shin_left
    60,  # j_heel_left
    42,  # j_thigh_right
    45,  # j_shin_right
    48,  # j_heel_right
    6,  # j_abdomen
    9,  # j_spine
    36,  # j_neck
    39,  # j_head
    12,  # j_scapula_left
    15,  # j_bicep_left
    18,  # j_forearm_left
    21,  # j_hand_left
    24,  # j_scapula_right
    27,  # j_bicep_right
    30,  # j_forearm_right
    33,  # j_hand_right
]


def axis2Euler(vec, offset_r=np.eye(3)):
    try:
        r = np.dot(offset_r, Rotation.from_rotvec(vec).as_matrix())
    except AttributeError:
        r = np.dot(offset_r, Rotation.from_rotvec(vec).as_dcm())

    try:
        r_euler = Rotation.from_matrix(r).as_euler('ZXY', True)
    except AttributeError:
        r_euler = Rotation.from_dcm(r).as_euler('ZXY', True)

    return r_euler


def dart2bvh(f_name: str, skel: pydart.Skeleton, qs, frame_rate: int):
    with open('/'.join(__file__.split('/')[:-1])+'/dart_skeleton.bvh', 'r') as f:
        skeleton_info = f.read()

    with open(f_name, 'w') as f:
        f.write(skeleton_info)
        f.write('MOTION\n')
        f.write(f'Frames: {len(qs)}\n')
        f.write(f'Frame Time: {1./frame_rate}\n')
        bvh_value = np.zeros(66)
        bvh_value[1] = 98.09
        bvh_value[2] = -3.08
        f.write(' '.join(map(str, bvh_value)) + '\n')

        for q in qs:
            bvh_value = np.zeros(66)
            joint: pydart.Joint
            for joint_idx, joint in enumerate(skel.joints):
                if joint.num_dofs() > 0:
                    dof_index = joint.dofs[0].index_in_skeleton()
                    bvh_index = dart2bvh_joint_index_map[joint_idx]
                    if joint.num_dofs() == 6:
                        bvh_value[0:3] = 100. * np.asarray(q[3:6]) + np.array((0., 98.09, -3.08))
                        bvh_value[3:6] = axis2Euler(q[0:3])
                    elif joint.num_dofs() == 3:
                        bvh_value[bvh_index:bvh_index+3] = axis2Euler(q[dof_index:dof_index+3])
                    elif joint.num_dofs() == 1:
                        bvh_value[bvh_index:bvh_index+3] = axis2Euler(q[dof_index] * joint.axis())
            f.write(' '.join(map(str, bvh_value)) +'\n')
