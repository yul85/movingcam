from PyCommon.modules.Motion.ysMotion import *
from PyCommon.modules.Motion.pmConstants import *
from copy import deepcopy


class PmLinearMotion(JointMotion):
    def __add__(self, nextMotion):
        """

        :param nextMotion:
        :type nextMotion: PmLinearMotionMotion
        :return: PmLinearMotion
        """
        if isinstance(nextMotion, PmLinearMotion):
            motion = PmLinearMotion(self.data)
            motion.data.extend(nextMotion.data)
            motion.fps = self.fps
            motion.resourceName = self.resourceName
            return motion
        else:
            raise TypeError

    def getSize(self):
        return len(self)

    def getPosture(self, i):
        """

        :param i:
        :return:
        :rtype: PmPosture
        """
        return self[i]

    def getMask(self):
        return self.getPosture(0).getMask()

    def getPositionVelocity(self, index, v):
        """

        :param i:
        :type i: int
        :param v:
        :type v: PmVector
        :return:
        """
        if index == 0:
            v.positionDifference(self.getPosture(index+1), self.getPosture(index))
        elif index == self.getSize() - 1:
            v.positionDifference(self.getPosture(index), self.getPosture(index-1))
        else:
            v.positionDifference(self.getPosture(index+1), self.getPosture(index-1))
            v = v/2.


class PmPosture(JointPosture):
    def __init__(self, skeleton):
        super(PmPosture, self).__init__(skeleton)
        self.skeleton = skeleton  # type: PmHuman

        # pmqm
        self.mask = MaskBit(PmHumanEnum.PELVIS)
        self.mask |= MaskBit(PmHumanEnum.CHEST)
        self.mask |= MaskBit(PmHumanEnum.NECK)
        self.mask |= MaskBit(PmHumanEnum.UPPER_RIGHT_ARM)
        self.mask |= MaskBit(PmHumanEnum.UPPER_LEFT_ARM)
        self.mask |= MaskBit(PmHumanEnum.LOWER_RIGHT_ARM)
        self.mask |= MaskBit(PmHumanEnum.LOWER_LEFT_ARM)
        self.mask |= MaskBit(PmHumanEnum.UPPER_RIGHT_LEG)
        self.mask |= MaskBit(PmHumanEnum.UPPER_LEFT_LEG)
        self.mask |= MaskBit(PmHumanEnum.LOWER_RIGHT_LEG)
        self.mask |= MaskBit(PmHumanEnum.LOWER_LEFT_LEG)
        self.mask |= MaskBit(PmHumanEnum.RIGHT_FOOT)
        self.mask |= MaskBit(PmHumanEnum.LEFT_FOOT)
        self.mask |= MaskBit(PmHumanEnum.RIGHT_TOE)
        self.mask |= MaskBit(PmHumanEnum.LEFT_TOE)

        self.pmqm_index_2_original_index = dict()
        self.pmqm_index_2_original_index[PmHumanEnum.PELVIS] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.PELVIS])
        self.pmqm_index_2_original_index[PmHumanEnum.CHEST] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.CHEST])
        self.pmqm_index_2_original_index[PmHumanEnum.NECK] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.NECK])
        self.pmqm_index_2_original_index[PmHumanEnum.UPPER_RIGHT_ARM] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.UPPER_RIGHT_ARM])
        self.pmqm_index_2_original_index[PmHumanEnum.UPPER_LEFT_ARM] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.UPPER_LEFT_ARM])
        self.pmqm_index_2_original_index[PmHumanEnum.LOWER_RIGHT_ARM] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.LOWER_RIGHT_ARM])
        self.pmqm_index_2_original_index[PmHumanEnum.LOWER_LEFT_ARM] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.LOWER_LEFT_ARM])
        self.pmqm_index_2_original_index[PmHumanEnum.UPPER_RIGHT_LEG] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.UPPER_RIGHT_LEG])
        self.pmqm_index_2_original_index[PmHumanEnum.UPPER_LEFT_LEG] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.UPPER_LEFT_LEG])
        self.pmqm_index_2_original_index[PmHumanEnum.LOWER_RIGHT_LEG] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.LOWER_RIGHT_LEG])
        self.pmqm_index_2_original_index[PmHumanEnum.LOWER_LEFT_LEG] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.LOWER_LEFT_LEG])
        self.pmqm_index_2_original_index[PmHumanEnum.RIGHT_FOOT] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.RIGHT_FOOT])
        self.pmqm_index_2_original_index[PmHumanEnum.LEFT_FOOT] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.LEFT_FOOT])
        self.pmqm_index_2_original_index[PmHumanEnum.RIGHT_TOE] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.RIGHT_TOE])
        self.pmqm_index_2_original_index[PmHumanEnum.LEFT_TOE] = self.skeleton.getElementIndex(human_part_name[PmHumanEnum.LEFT_TOE])


    def getIndexFromPmqmIndex(self, i):
        index = self.pmqm_index_2_original_index[i]
        assert index is not None, human_part_name[i] + ' is not in this file'
        return index

    def getGlobalTransf(self, i):
        return self.globalTs[self.getIndexFromPmqmIndex(i)]

    def getGlobalPosition(self, i):
        return mm.T2p(self.globalTs[self.getIndexFromPmqmIndex(i)])

    def getGlobalTranslation(self, i):
        return mm.T2p(self.globalTs[self.getIndexFromPmqmIndex(i)])

    def getTransf(self, i):
        assert 0 <= i < PM_HUMAN_NUM_LINKS
        if i == PmHumanEnum.PELVIS:
            return self.globalTs[0]
        else:
            return mm.SO3ToSE3(self.localRs[self.getIndexFromPmqmIndex(i)])

    def getMask(self):
        return self.mask


class PmHuman(JointSkeleton):
    pass


class PmVectorArray:
    def __init__(self, joint_num):
        self.data = []  # type: list[PmVector]
        self.joint_num = joint_num

    def setSize(self, size):
        for i in range(size):
            self.data.append(PmVector(self.joint_num))

    def setVector(self, j, v):
        self.data[j] = deepcopy(v)


class PmVector:
    def __init__(self, joint_num):
        # pmqm
        self.mask = MaskBit(PmHumanEnum.PELVIS)
        self.mask |= MaskBit(PmHumanEnum.CHEST)
        self.mask |= MaskBit(PmHumanEnum.NECK)
        self.mask |= MaskBit(PmHumanEnum.UPPER_RIGHT_ARM)
        self.mask |= MaskBit(PmHumanEnum.UPPER_LEFT_ARM)
        self.mask |= MaskBit(PmHumanEnum.LOWER_RIGHT_ARM)
        self.mask |= MaskBit(PmHumanEnum.LOWER_LEFT_ARM)
        self.mask |= MaskBit(PmHumanEnum.UPPER_RIGHT_LEG)
        self.mask |= MaskBit(PmHumanEnum.UPPER_LEFT_LEG)
        self.mask |= MaskBit(PmHumanEnum.LOWER_RIGHT_LEG)
        self.mask |= MaskBit(PmHumanEnum.LOWER_LEFT_LEG)
        self.mask |= MaskBit(PmHumanEnum.RIGHT_FOOT)
        self.mask |= MaskBit(PmHumanEnum.LEFT_FOOT)
        self.mask |= MaskBit(PmHumanEnum.RIGHT_TOE)
        self.mask |= MaskBit(PmHumanEnum.LEFT_TOE)

        self.linear = np.zeros(3)
        self.angular = [np.zeros(3) for i in range(PM_HUMAN_NUM_LINKS)]
        self.joint_num = joint_num

    def __truediv__(self, other):
        if isinstance(other, float):
            v = PmVector(self.joint_num)
            v.linear = self.linear / other
            for i in range(len(self.angular)):
                v.angular = self.angular[i] / other
            return v

    def getMask(self):
        return self.mask

    def positionDifference(self, p1, p2):
        """

        :param p1:
        :type p1: PmPosture
        :param p2:
        :type p2: PmPosture
        :return:
        """
        self.mask = p1.getMask() & p2.getMask()

        t1 = mm.PlaneProject(p1.getTransf(0))
        t2 = mm.PlaneProject(p2.getTransf(0))

        # calib = mm.invertSE3(t2) * t1
        calib = t1 * mm.invertSE3(t2)

        self.setLinearVector(np.zeros(3))

        for i in range(PM_HUMAN_NUM_LINKS):
            if self.getMask() & MaskBit(i):
                a1 = p1.getGlobalTranslation(i)
                # a2 = (np.zeros(3) + p2.getGlobalTranslation(i)) * calib
                a2 = mm.affine_pos(calib, p2.getGlobalTranslation(i))

                self.setAngularVector(i, a2 - a1)

        return self

    def setLinearVector(self, v):
        self.linear = v.copy()
        # self.mask |= MaskBit(0)

        return v

    def getLinearVector(self):
        return self.linear

    def setAngularVector(self, i, v):
        assert 0 <= i < PM_HUMAN_NUM_LINKS

        self.angular[i] = v.copy()
        # self.mask |= MaskBit(i)

        return v