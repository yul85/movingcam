import copy
import math
import numpy as np
from PyCommon.modules.Math import mmMath as mm

from scipy.spatial.transform.rotation import Rotation

from scipy.interpolate import BSpline


class DartSkelMotion(object):
    def __init__(self):
        self.qs = []
        self.dqs = []
        self.fps = 30.
        self.has_loop = False
        self.loop = [0, 0]

    def __len__(self):
        assert len(self.qs) == len(self.dqs)
        return len(self.qs)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.fps = 1./float(f.readline().split(' ')[-1])
            for s in f.read().splitlines():
                ss = s.replace('[', '').split(']')
                sq, sdq = list(map(float, ss[0].split(','))), list(map(float, ss[1].split(',')))
                self.qs.append(np.asarray(sq))

                self.dqs.append(np.asarray(sdq))

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('Frame Times: '+str(1./self.fps))
            f.write('\n')
            for frame in range(len(self.qs)):
                f.write(str([d for d in np.asarray(self.qs[frame])]))
                f.write(str([d for d in np.asarray(self.dqs[frame])]))
                f.write('\n')

    def append(self, _q, _dq):
        self.qs.append(copy.deepcopy(_q))
        self.dqs.append(copy.deepcopy(_dq))

    def extend(self, frame, _qs, _dqs):
        qs = copy.deepcopy(_qs)
        dqs = copy.deepcopy(_dqs)
        offset_x = self.qs[frame][3] - qs[0][3]
        offset_z = self.qs[frame][5] - qs[0][5]
        del self.qs[frame:]
        del self.dqs[frame:]

        self.qs.extend(qs)
        self.dqs.extend(dqs)

        for i in range(len(qs)):
            self.qs[frame+i][3] += offset_x
            self.qs[frame+i][5] += offset_z

    def rotate_by_offset(self, R_offset, origin):
        for i in range(len(self.qs)):
            self.qs[i][0:3] = mm.logSO3(np.dot(R_offset, mm.exp(self.qs[i][:3])))
            self.qs[i][3:6] = np.dot(R_offset, self.qs[i][3:6] - origin) + origin

    def translate_by_offset(self, offset):
        for i in range(len(self.qs)):
            self.qs[i][3:6] += offset

    def set_q(self, frame, q, dq):
        self.qs[frame] = q
        self.dqs[frame] = dq

    def get_q(self, frame):
        assert self.has_loop or frame < len(self.qs)

        if self.has_loop and frame >= len(self.qs):
            loop_len = self.loop[1] + 1 - self.loop[0]
            loop_idx_offset = self.loop[0]
            _frame = int((frame - loop_idx_offset) % loop_len + loop_idx_offset)
            i = int((frame-loop_idx_offset)//loop_len)

            offset = self.qs[self.loop[1]][3:6] - self.qs[self.loop[0]][3:6]
            offset[1] = 0.

            q = copy.deepcopy(self.qs[_frame])
            q[3:6] += i * offset

            return q
        else:
            return self.qs[frame]

    def get_dq(self, frame):
        assert self.has_loop or frame < len(self.dqs)

        if self.has_loop:
            return self.dqs[self.get_frame_looped(frame)]
        else:
            return self.dqs[frame]

    def reset_root_trajectory_backflip_a(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            self.qs[frame][0] = -0.39
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.07
            self.qs[frame][3] = 0.
            self.qs[frame][4] = -0.13
            self.qs[frame][5] = 0.

    def reset_root_trajectory_cartwheel_b(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            skel.set_positions(self.qs[frame])
            self.qs[frame][0] = -0.35
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.
            self.qs[frame][3] = 0.
            self.qs[frame][4] = -0.03
            self.qs[frame][5] = 0.

    def reset_root_trajectory_dance(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            self.qs[frame][0] = -0.23
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.1
            self.qs[frame][3] = 0.
            self.qs[frame][4] = -0.1
            self.qs[frame][5] = 0.

    def reset_root_trajectory_high_jump(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            skel.set_positions(self.qs[frame])
            self.qs[frame][0] = -0.5
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.
            self.qs[frame][4] = -0.05
            self.qs[frame][5] = 0.

    def reset_root_trajectory_hurdle(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            self.qs[frame][0] = -0.3
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.
            self.qs[frame][3] = 0.
            self.qs[frame][4] = -0.09
            self.qs[frame][5] = 0.

    def reset_root_trajectory_jump(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            self.qs[frame][0] = -0.2
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.0
            self.qs[frame][3] = 0.
            self.qs[frame][4] = -0.02
            self.qs[frame][5] = 0.

    def reset_root_trajectory_lateral_jump(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            self.qs[frame][0] = -0.35
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.
            self.qs[frame][3] = 0.
            self.qs[frame][4] = -0.05
            self.qs[frame][5] = 0.

    def reset_root_trajectory_parkour1(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            skel.set_positions(self.qs[frame])
            self.qs[frame][0] = -0.7
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.
            self.qs[frame][3] = 0.
            self.qs[frame][4] = -0.
            self.qs[frame][5] = 0.

            if frame == 23 or frame == 24:
                self.qs[frame][13] = 0.75
                self.qs[frame][6] = -0.05

    def reset_root_trajectory_parkour2(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            skel.set_positions(self.qs[frame])
            self.qs[frame][0] = -0.7
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.
            self.qs[frame][3] = 0.
            self.qs[frame][4] = -0.09
            self.qs[frame][5] = 0.

    def reset_root_trajectory_parkour3(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            self.qs[frame][0] = -0.7
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.
            self.qs[frame][3] = 0.
            self.qs[frame][4] = 0.01
            self.qs[frame][5] = 0.

    def reset_root_trajectory_parkour3_optimize(self, skel):
        frame_num = len(self.qs)

        for frame in range(frame_num):
            self.qs[frame][0] = -0.7
            self.qs[frame][1] = 0.
            self.qs[frame][2] = 0.
            self.qs[frame][3] = 0.
            self.qs[frame][4] = 0.01
            self.qs[frame][5] = 0.

            if frame == 14:
                self.qs[frame][13] = 0.35
                self.qs[frame][6] = -1.25

    def refine_dqs(self, skel, start_frame=0):
        for frame in range(start_frame, len(self.dqs)):
            if frame == len(self.dqs)-1:
                self.dqs[frame] = np.asarray(skel.position_differences(self.qs[frame], self.qs[frame-1])) * self.fps
            else:
                self.dqs[frame] = np.asarray(skel.position_differences(self.qs[frame+1], self.qs[frame])) * self.fps

    def set_loop(self, begin_frame, end_frame):
        assert(begin_frame >= 0 and end_frame < len(self.qs) and begin_frame < end_frame)
        self.has_loop = True
        self.loop[0], self.loop[1] = begin_frame, end_frame

    def get_frame_looped(self, frame):
        assert self.has_loop or frame < len(self.qs)

        if frame < len(self.qs):
            return frame
        else:
            loop_len = self.loop[1] + 1 - self.loop[0]
            loop_offset = self.loop[0]
            return int((frame - loop_offset) % loop_len + loop_offset)

    def interpolate(self, init_q, duration, first=True):
        """
        interpolate init_q to first or last of motion for duration
        :param init_q:
        :param duration:
        :param first:
        :return:
        """
        assert len(init_q) == len(self.qs[0])

        dof = len(init_q)
        if first:
            for i in range(duration):
                for dof_idx in range(0, dof, 3):
                    if dof_idx != 3:
                        R1, R2 = map(mm.exp, [init_q[dof_idx:dof_idx+3], self.qs[i][dof_idx:dof_idx+3]])
                        self.qs[i][dof_idx:dof_idx+3] = mm.logSO3(mm.slerp(R1, R2, (i+1)/duration))

        else:
            raise NotImplementedError

    def connect_to(self, init_q, interpolate_duration):
        """
        connect to other pose

        :param init_q:
        :param interpolate_duration:
        :return:
        """
        assert len(init_q) == len(self.qs[0])

        # align trajectory first
        t_offset = init_q[3:6] - self.qs[0][3:6]
        t_offset[1] = 0.
        self.translate_by_offset(t_offset)

        R_offset = mm.getSO3FromVectors(
                    np.multiply(np.array([1., 0., 1.]), np.dot(mm.exp(self.qs[0][0:3]), mm.unitX())),
                    np.multiply(np.array([1., 0., 1.]), np.dot(mm.exp(init_q[0:3]), mm.unitX()))
        )
        self.rotate_by_offset(R_offset, init_q[3:6])

        dof = len(init_q)
        for i in range(interpolate_duration-1):
            for dof_idx in range(0, dof, 3):
                if dof_idx != 3:
                    R1, R2 = map(mm.exp, [init_q[dof_idx:dof_idx+3], self.qs[i][dof_idx:dof_idx+3]])
                    self.qs[i][dof_idx:dof_idx+3] = mm.logSO3(mm.slerp(R1, R2, (i+1)/interpolate_duration))

    def set_avg_x_vel(self, vel_x):
        avg_vel_x = (self.qs[-1][3] - self.qs[0][3]) / len(self) * self.fps
        for i in range(1, len(self)):
            self.qs[i][3] += (vel_x - avg_vel_x) * i / self.fps

    @staticmethod
    def q_dq_mirror_to_xy(_q, _dq, skel):
        """

        :type skel: pydart2.Skeleton
        :param skel:
        :return:
        """
        mirror_T = np.eye(3)
        mirror_T[2, 2] = -1.

        def mirror_axis_angle(_axis_angle):
            return mm.logSO3(np.dot(np.dot(mirror_T, mm.exp(_axis_angle)), mirror_T))

        def mirror_ang_vel(_ang_vel):
            return np.array([-_ang_vel[0], -_ang_vel[1], _ang_vel[2]])

        dof = len(_q)
        q = np.zeros_like(_q)
        dq = np.zeros_like(_dq)
        q[3:5] = _q[3:5]
        q[5] = -_q[5]
        dq[3:5] = _dq[3:5]
        dq[5] = -_dq[5]
        for dof_idx in range(0, dof, 3):
            if dof_idx != 3:
                dof_name = skel.dof(dof_idx).name
                mirror_dof_name = dof_name
                if 'left' in mirror_dof_name:
                    mirror_dof_name = dof_name.replace('left', 'right')
                elif 'right' in mirror_dof_name:
                    mirror_dof_name = dof_name.replace('right', 'left')
                mirror_dof_idx = skel.dof(mirror_dof_name).index_in_skeleton()
                q[mirror_dof_idx:mirror_dof_idx+3] = mirror_ang_vel(_q[dof_idx:dof_idx+3])
                dq[mirror_dof_idx:mirror_dof_idx+3] = mirror_ang_vel(_dq[dof_idx:dof_idx+3])

        return q, dq

    def mirror_to_xy(self, skel):
        """

        :type skel: pydart2.Skeleton
        :param skel:
        :return:
        """
        mirror_T = np.eye(3)
        mirror_T[2, 2] = -1.

        def mirror_ang_vel(_ang_vel):
            return np.array([-_ang_vel[0], -_ang_vel[1], _ang_vel[2]])

        qs_new = []
        dqs_new = []
        dof = len(self.qs[0])
        for i in range(len(self)):
            q = np.zeros_like(self.qs[i])
            dq = np.zeros_like(self.dqs[i])
            q[3:5] = self.qs[i][3:5]
            q[5] = -self.qs[i][5]
            dq[3:5] = self.dqs[i][3:5]
            dq[5] = -self.dqs[i][5]
            for dof_idx in range(0, dof, 3):
                if dof_idx != 3:
                    dof_name = skel.dof(dof_idx).name
                    mirror_dof_name = dof_name
                    if 'left' in mirror_dof_name:
                        mirror_dof_name = dof_name.replace('left', 'right')
                    elif 'right' in mirror_dof_name:
                        mirror_dof_name = dof_name.replace('right', 'left')
                    mirror_dof_idx = skel.dof(mirror_dof_name).index_in_skeleton()
                    q[mirror_dof_idx:mirror_dof_idx+3] = mirror_ang_vel(self.qs[i][dof_idx:dof_idx+3])
                    dq[mirror_dof_idx:mirror_dof_idx+3] = mirror_ang_vel(self.dqs[i][dof_idx:dof_idx+3])

            qs_new.append(q)
            dqs_new.append(dq)
        del self.qs[:]
        del self.dqs[:]
        self.qs.extend(qs_new)
        self.dqs.extend(dqs_new)

    def get_q_by_time(self, time):
        frame0 = int(time * self.fps)
        frame1 = int(time * self.fps) + 1
        q0 = self.get_q(frame0)
        q1 = self.get_q(frame1)
        return self.slerp(q0, q1, time*self.fps - frame0)

    @staticmethod
    def slerp(q_0, q_1, t):
        """
        slerp skel pose.
        return q_0 when t=0, return q_1 when t=1.

        :param q_0:
        :param q_1:
        :param t:
        :return:
        """
        assert len(q_0) == len(q_1)

        q = np.zeros_like(q_0)
        dof = len(q_0)
        for dof_idx in range(0, dof, 3):
            if dof_idx != 3:
                R1, R2 = map(mm.exp, [q_0[dof_idx:dof_idx+3], q_1[dof_idx:dof_idx+3]])
                q[dof_idx:dof_idx+3] = mm.logSO3(mm.slerp(R1, R2, t))
            else:
                q[dof_idx:dof_idx+3] = (1.-t) * q_0[dof_idx:dof_idx+3] + t * q_1[dof_idx:dof_idx+3]

        return q


def axis2Euler(vec, offset_r=np.eye(3)):
    r = np.dot(offset_r, Rotation.from_rotvec(vec).as_dcm())
    r_after = np.dot(np.dot(mm.rotY(-math.pi/2.), r), mm.rotY(-math.pi/2.).T)
    return Rotation.from_dcm(r_after).as_euler('ZXY', True)


def skelqs2bvh(f_name, skel, qs):
    #make bvh file

    # 0:6         #pelvis
    # 15, 16, 17 # right leg
    # 18, 19, 20
    # 21, 22, 23
    # zero for toes
    # 24, 25, 26  #spine
    # 27, 28, 29
    # 30, 31, 32
    # 33, 34, 35, # left arm
    # 36, 37, 38
    # 39, 40, 41
    # 42, 43, 44
    # 45, 46, 47  #right arm
    # 48, 49, 50
    # 51, 52, 53
    # 54, 55, 56
    # 6, 7, 8     #left leg
    # 9, 10, 11
    # 12, 13, 14
    # zero for toes

    with open('/'.join(__file__.split('/')[:-1])+'/skeleton.bvh', 'r') as f:
        skeleton_info = f.read()

    with open(f_name, 'w') as f:
        f.write(skeleton_info)
        f.write('MOTION\r\n')
        f.write('Frames: '+str(len(qs))+'\r\n')
        f.write('Frame Time: 0.0333333\r\n')
        t_pose_angles = [0. for _ in range(len(qs[0])+6)]
        t_pose_angles[1] = 104.721
        f.write(' '.join(map(str, t_pose_angles))+'\r\n')

        for q in qs:
            euler_middle_q = np.asarray(skel.q)
            for joit_i in range(int(len(q) / 3)):
                if joit_i != 1:
                    temp_axis_angle = np.asarray([q[3*joit_i], q[3*joit_i+1], q[3*joit_i+2]])
                    r_offset = np.eye(3)
                    if 'scapular_left' in skel.dof(3*joit_i).name:
                        r_offset = mm.rotX(-0.9423)
                    elif 'scapular_right' in skel.dof(3*joit_i).name:
                        r_offset = mm.rotX(0.9423)
                    elif 'bicep_left' in skel.dof(3*joit_i).name:
                        r_offset = mm.rotX(-1.2423)
                    elif 'bicep_right' in skel.dof(3*joit_i).name:
                        r_offset = mm.rotX(1.2423)
                    euler_result = axis2Euler(temp_axis_angle, r_offset)
                    euler_middle_q[3*joit_i:3*joit_i+3] = euler_result

            euler_q = np.zeros(len(q)+6)       # add two toes dofs (6 = 3x2)
            euler_q[0:3] = np.dot(mm.rotY(-math.pi / 2.), q[3:6] * 100.) + np.array([0., 104.721, 0.])
            euler_q[3:6] = euler_middle_q[0:3]
            euler_q[6:15] = euler_middle_q[15:24]
            euler_q[15:18] = np.zeros(3)
            euler_q[18:51] = euler_middle_q[24:]
            euler_q[51:60] = euler_middle_q[6:15]
            euler_q[60:63] = np.zeros(3)

            f.write(' '.join(map(str, euler_q)))
            f.write('\r\n')


if __name__ == '__main__':
    # mo = DartSkelMotion()
    # mo.load('skate.skmo')
    skelqs2bvh('test.bvh', None, [])
