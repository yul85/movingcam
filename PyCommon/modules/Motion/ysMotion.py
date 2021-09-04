import copy
import math
import numpy as np
import operator as op

import sys
# if '..' not in sys.path:
#     sys.path.append('..')
# import Math.mmMath as mm
# import Math.csMath as cm
# import Util.ysPythonEx as ype
from PyCommon.modules.Math import mmMath as mm
# from PyCommon.modules.Math import csMath as cm
from PyCommon.modules.Util import ysPythonEx as ype

# ===============================================================================
# motion classes
# ===============================================================================
class MotionSystem:
    def __init__(self):
        self.motions = []

    def addMotion(self, motion):
        self.motions.append(motion)

    def removeAllMotions(self):
        del self.motions[:]

    def updateFrame(self, frame):
        for i in range(len(self.motions)):
            self.motions[i].goToFrame(frame)

    def getNumMotion(self):
        return len(self.motions)

    def getMaxFrame(self):
        maxFrame = 0
        for motion in self.motions:
            if maxFrame < len(motion)-1:
                maxFrame = len(motion)-1
        return maxFrame

            
#===============================================================================
# base classes
#===============================================================================
# class Motion(list):
#     def __init__(self, ls=[]):
#         """
#
#         :param ls: list[Posture]
#         """
#         list.__init__(self, ls)
#         self.data = []
#         self.frame = 0
#         self.fps = 30.
#         self.resourceName = 'unnamed'
#
#     def __getslice__(self, i, j):
#         motion = self.__new__(self.__class__)
#         motion.__init__(list.__getslice__(self, i, j))
#         motion.fps = self.fps
#         motion.resourceName = self.resourceName
#         return motion
#     def __add__(self, nextMotion):
#         motion = self.__new__(self.__class__)
#         motion.__init__(list.__add__(self, nextMotion))
#         motion.fps = self.fps
#         motion.resourceName = self.resourceName
#         return motion
class Motion:
    def __init__(self, ls=None):
        """

        :param ls: list[Posture]
        """
        # list.__init__(self, ls)
        self.data = []  # type: list[Posture]
        if ls is not None:
            for l in ls:
                self.data.append(l)

        self.frame = 0  # type: int
        self.fps = 30.  # type: float
        self.resourceName = 'unnamed'  # type: str

    # def __getslice__(self, i, j):
    #     motion = Motion(self.data[i:j])
    #     motion.fps = self.fps
    #     motion.resourceName = self.resourceName
    #     return motion

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        elif isinstance(item, slice):
            motion = self.__class__(self.data[item])
            motion.fps = self.fps
            motion.resourceName = self.resourceName
            return motion
        else:
            raise TypeError

    def __delitem__(self, key):
        del self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, nextMotion):
        """

        :param nextMotion: Motion
        :return: Motion
        """
        if isinstance(nextMotion, Motion):
            motion = Motion(self.data)
            motion.data.extend(nextMotion.data)
            motion.fps = self.fps
            motion.resourceName = self.resourceName
            return motion
        else:
            raise TypeError

    def __len__(self):
        return len(self.data)

    def append(self, posture):
        if isinstance(posture, Posture):
            self.data.append(posture)
        else:
            raise TypeError

    def extend(self, postures):
        if isinstance(postures, Motion) or isinstance(postures, list):
            self.data.extend(postures)
        else:
            raise TypeError

    def getState(self):
        return self.frame

    def setState(self, state):
        self.frame = state

    def getPostureAt(self, frame):
        floor = int(frame)
        t = frame - floor
        if t==0.0:
            return self[floor]
        else:
            return self[floor].blendPosture(self[floor+1], t)
        
    def getPosition(self, index, frame):
        return self.getPostureAt(frame).getPosition(index)

    def getPositions(self, frame):
        return self.getPostureAt(frame).getPositions()

    def getVelocity(self, index, frame0, frame1=None):
        return self._getDerivative(index, frame0, frame1, self.getPosition, op.sub)

    def getVelocities(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getPositions, op.sub)

    def getAcceleration(self, index, frame0, frame1=None):
        return self._getDerivative(index, frame0, frame1, self.getVelocity, op.sub)

    def getAccelerations(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getVelocities, op.sub)

    def getFiniteDifferenceFrames(self, frame):
        prevFrame = frame-1 if frame > 0 else frame
        nextFrame = frame+1 if frame < len(self)-1 else frame
        return prevFrame, nextFrame

    def _getDerivativeBasic(self, frame0, frame1, positionFunc, subFunc):
        if frame0 == frame1 or len(self) == 1:
            return mm.O_Vec3()

        if frame1 is None:
            frame0, frame1 = self.getFiniteDifferenceFrames(frame0)
        p0 = positionFunc(frame0)
        p1 = positionFunc(frame1)
        return (self.fps/(frame1-frame0)) * subFunc(p1, p0)

    def _getDerivative(self, index, frame0, frame1, positionFunc, subFunc):
        if frame0 == frame1 or len(self) == 1:
            return mm.O_Vec3()
        
        if frame1 is None:
            frame0, frame1 = self.getFiniteDifferenceFrames(frame0)
        p0 = positionFunc(index, frame0)
        p1 = positionFunc(index, frame1)
        return (self.fps/(frame1-frame0)) * subFunc(p1, p0)

    def _getDerivatives(self, frame0, frame1, positionsFunc, subFunc):
        if frame0 == frame1 or len(self) == 1:
            return [mm.O_Vec3()]*self[0].skeleton.getElementNum()
        
        if frame1 is None:
            frame0, frame1 = self.getFiniteDifferenceFrames(frame0)
        positions0 = positionsFunc(frame0)
        positions1 = positionsFunc(frame1)
        return list(map(lambda p0, p1: (self.fps/(frame1-frame0)) * subFunc(p1, p0), positions0, positions1))

    def goToFrame(self, frame):
        # if frame > -1 and frame < len(self):
        if -1 < frame < len(self):
            self.frame = frame

    def translateByOffset(self, offset, update=True):
        for posture in self:
            posture.translateByOffset(offset, update)

    def rotateByOffset(self, R_offset, update=True):
        for posture in self:
            posture.rotateByOffset(R_offset, update)

    def copy(self):
        copied = self.__class__()
        # copied = self.__new__(self.__class__)
        # copied.__init__([None]*len(self))
        copied.fps = self.fps
        for i in range(len(self)):
            copied.append(self[i].copy())
        return copied

    def scale(self, scale, update=True):
        self[0].skeleton.scale(scale)
        for posture in self:
            posture.scale(scale, update)


class Skeleton:
    def __init__(self):
        self.elements = []
        self.elementNames = []  # type: list[str]
        self.reverseElementNames = {}

    def __str__(self):
        s = ''
#        s += self.elementNames.__str__() + '\n'
        s += '<ELEMENTS>\n'
        for i in range(len(self.elementNames)):
            s += '[%d]:\'%s\', ' % (i, self.elementNames[i])
        s += '\n'
        return s

    def addElement(self, element, name):
        self.elements.append(element)
        if name is not None:
            self.elementNames.append(name)
            self.reverseElementNames[name] = len(self.elementNames)-1
        
    def getElement(self, index):
        return self.elements[index]

    def getElementNum(self):
        return len(self.elements)

    def getElementName(self, index):
        if index < len(self.elementNames):
            return self.elementNames[index]
        else:
            return None

    def getElementIndex(self, name):
        if name in self.reverseElementNames:
            return self.reverseElementNames[name]
        else:
            return None

    def removeElement(self, index):
        del self.elements[index:index+1]
        del self.elementNames[index:index+1]
        
        self.reverseElementNames.clear()
        for i in range(len(self.elements)):
            self.reverseElementNames[self.elementNames[i]] = i

    def scale(self, scale):
        raise NotImplementedError


class Posture:
    def __init__(self, skeleton):
        self.skeleton = skeleton  # type: Skeleton

    def getPosition(self, index):
        raise NotImplementedError

    def getPositions(self):
        raise NotImplementedError

    def blendPosture(self, otherPosture, t):
        raise NotImplementedError

    def translateByOffset(self, offset):
        raise NotImplementedError

    def scale(self, scale):
        raise NotImplementedError


#===============================================================================
# point motion
#===============================================================================
class PointSkeleton(Skeleton):
    pass


class PointPosture(Posture):
    def __init__(self, pointSkeleton):
        Posture.__init__(self, pointSkeleton)
        self.positions = [None]*pointSkeleton.getElementNum()

    def setPosition(self, index, position):
        self.positions[index] = position

    def getPosition(self, index):
        return self.positions[index]

    def getPositions(self):
        return copy.copy(self.positions)

    def blendPosture(self, otherPosture, t):
        p = PointPosture(self.skeleton)
        for i in range(len(self.positions)):
            p.positions[i] = mm.linearInterpol(self.positions[i], otherPosture.positions[i], t)
        return p

    def translateByOffset(self, offset, update=True):
        for i in range(len(self.positions)):
            self.positions[i] += offset


#===============================================================================
# joint motion
#
# number of links: n <-> number of joints: n (including imaginary root joint)
#                    <-> number of internal joints: n-1
# 
# parent        <--->        child
# joint[0] - link[0] - joint[1] - link[1] - ... - joint[n-1] - link[n-1]
#
# joint[0]: (imaginary root joint)
# link[0]: (root body) 
#===============================================================================
class JointMotion(Motion):
    def __init__(self, ls=None):
        """

        :param ls:
        :type ls: list[JointPosture]
        """
        # Motion.__init__(self, None)
        super(JointMotion, self).__init__(ls)
        self.data = []  # type: list[JointPosture]
        if ls is not None:
            for l in ls:
                self.data.append(l)
        self.frame = 0
        self.fps = 30.
        self.resourceName = 'unnamed'

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        elif isinstance(item, slice):
            motion = self.__class__(self.data[item])
            motion.fps = self.fps
            motion.resourceName = self.resourceName
            return motion
        else:
            raise TypeError

    def __add__(self, nextMotion):
        """

        :param nextMotion: Motion
        :return: Motion
        """
        if isinstance(nextMotion, Motion):
            motion = JointMotion(self.data)
            motion.data.extend(nextMotion.data)
            motion.fps = self.fps
            motion.resourceName = self.resourceName
            return motion
        else:
            raise TypeError

    # lv : linear velocity, av : angular velocity, la : linear acceleration, aa : angular acceleration
    # p: Vec3(position), R : SO3(orientation)
    # _g : w.r.t. global frame, _l : w.r.t. local frame
    # x[0] : x of joint[0]

    def get_q(self, frame):
        '''
        get generalized position in joint coordinate

        linear second and linear part is on world frame
        :param frame:
        :return:
        '''
        return self[frame].get_q()

    def get_q_by_time(self, t):
        """
        get generalized position in joint coordinate in continuous time

        linear second and linear part is on world frame
        :param t:
        :return:
        """
        frame0 = int(t * self.fps)
        frame1 = frame0 + 1
        if t <= 0:
            return self[0].get_q()

        if frame1 >= len(self):
            return self[-1].get_q()

        return self[frame0].blendPosture(self[frame1], t * self.fps - frame0).get_q()

    def get_dq(self, frame0, frame1=None):
        """
        get generalized velocity in joint coordinate

        linear second and linear part is dependent on root body frame
        :param frame0:
        :param frame1:
        :return:
        """
        return self._getDerivativeBasic(frame0, frame1, self.get_q, lambda x, y: x-y)

    def get_dq_dart(self, frame0, frame1=None):
        """
        get generalized velocity in joint coordinate

        linear second and linear part is dependent on root body frame
        :param frame0:
        :param frame1:
        :return:
        """
        # dof_vel = self.getDOFVelocitiesLocal(frame0)
        # temp = dof_vel[0][0:3].copy()
        # dof_vel[0][0:3] = dof_vel[0][3:6]
        # dof_vel[0][3:6] = temp
        # return np.concatenate(dof_vel)
        return np.concatenate([np.concatenate( (self.getJointAngVelocityLocal(0, frame0), np.dot(self[frame0].getLocalR(0).T, self.getJointVelocityGlobal(0, frame0))) )] \
               + self.getInternalJointAngVelocitiesLocal(frame0))


    def get_dq_dart_by_time(self, t):
        """
        get generalized velocity in joint coordinate in continuous time

        linear second and linear part is dependent on root body frame
        :param t:
        :return:
        """
        frame0 = int(t * self.fps)
        frame1 = frame0 + 1
        if t <= 0:
            return self.get_dq_dart(0)

        if frame1 >= len(self):
            return self.get_dq_dart(len(self)-1)
        dt = t * self.fps - frame0
        return (1.-dt) * self.get_dq_dart(frame0) + dt * self.get_dq_dart(frame1)

    def get_ddq(self, frame0, frame1=None):
        '''
        get generalized acceleration in joint coordinate

        linear first and linear part is independent on root body
        :param frame0:
        :param frame1:
        :return:
        '''
        return self._getDerivativeBasic(frame0, frame1, self.get_dq, lambda x, y: x-y)

    def getDOFPositions(self, frame):
        '''
        [(p_g[0], R_g[0]), R_l[1], R_l[2], ... ,R_l[n-1]]

        :param frame:
        :return:
        '''
        # return [self[frame].getGlobalT(0)] + self.getInternalJointOrientationsLocal(frame)
        return [(self[frame].rootPos, self[frame].getLocalR(0))] + self.getInternalJointOrientationsLocal(frame)
    
    # [(p_l[0], R_l[0]), R_l[1], R_l[2], ... ,R_l[n-1]]
    def getDOFPositionsLocal(self, frame):
        # return [self[frame].getGlobalT(0)] + self.getInternalJointOrientationsLocal(frame)
        return [(np.dot(self[frame].getLocalR(0).T, self[frame].rootPos), self[frame].getLocalR(0))] + self.getInternalJointOrientationsLocal(frame)

    def getDOFPositionsEuler(self, frame):
        positions = self.getDOFPositions(frame)
        positionsEuler = []
        positionsEuler.append(np.concatenate((positions[0][0], mm.R2ZXY(positions[0][1]))))
        for i in range(1,len(positions)): 
            positionsEuler.append(mm.R2ZXY(positions[i]))
        return positionsEuler

    # [lv_g[0]<hmerge>av_l[0], av_l[1], av_l[2], ... av_l[n-1]]
    def getDOFVelocities(self, frame):
        # return [np.concatenate( (self.getJointVelocityGlobal(0, frame), self.getJointAngVelocityGlobal(0, frame)) )]\
        #        + self.getInternalJointAngVelocitiesLocal(frame)
        return [np.concatenate( (self.getJointVelocityGlobal(0, frame), self.getJointAngVelocityLocal(0, frame)) )]\
                + self.getInternalJointAngVelocitiesLocal(frame)
                
    # [lv_l[0]<hmerge>av_l[0], av_l[1], av_l[2], ... av_l[n-1]]
    def getDOFVelocitiesLocal(self, frame):
        return [np.concatenate( (np.dot(self[frame].getLocalR(0).T, self.getJointVelocityGlobal(0, frame)), self.getJointAngVelocityLocal(0, frame)) )]\
                + self.getInternalJointAngVelocitiesLocal(frame)

    def getDOFVelocitiesEuler(self, frame):
        # if frame == 0:
        #     frame = frame+1
        # positionEuler0 = self.getDOFPositionsEuler(frame-1)
        # positionEuler1 = self.getDOFPositionsEuler(frame)
        # position0 = self.getDOFVelocities(frame)
        # dotR = np.dot(self[frame].getLocalR(0), self[frame].getLocalR(0))
        # velocities = []
        # for i in range(0,len(position0)):
        #     velocities.append(30.*(positionEuler1[i]-positionEuler0[i]))
        velocities = self.getDOFVelocities(frame)
        velocityRoot = velocities[0].copy()
        velocities[0][3] = velocityRoot[5] #Z
        velocities[0][4] = velocityRoot[3] #X
        velocities[0][5] = velocityRoot[4] #Y
        for i in range(1, len(velocities)):
            velocity = velocities[i].copy()
            velocities[i][0] = velocity[2]
            velocities[i][1] = velocity[0]
            velocities[i][2] = velocity[1]
        return velocities

    # [la_g[0]<hmerge>aa_l[0], aa_l[1], aa_l[2], ... aa_l[n-1]] 
    def getDOFAccelerations(self, frame):
        # return [np.concatenate( (self.getJointAccelerationGlobal(0, frame), self.getJointAngAccelerationGlobal(0, frame)) )]\
        #        + self.getInternalJointAngAccelerationsLocal(frame)
        return [np.concatenate( (self.getJointAccelerationGlobal(0, frame), self.getJointAngAccelerationLocal(0, frame)) )]\
                + self.getInternalJointAngAccelerationsLocal(frame)
                 
    # [la_l[0]<hmerge>aa_l[0], aa_l[1], aa_l[2], ... aa_l[n-1]] 
    def getDOFAccelerationsLocal(self, frame):
        return [np.concatenate( (np.dot(self[frame].getLocalR(0).T, self.getJointAccelerationGlobal(0, frame)), self.getJointAngAccelerationLocal(0, frame)) )]\
                + self.getInternalJointAngAccelerationsLocal(frame)

    def getDOFAccelerationsEuler(self, frame):
        # if frame == 0:
        #     frame = frame+1
        # velocity0 = self.getDOFVelocitiesEuler(frame-1)
        # velocity1 = self.getDOFVelocitiesEuler(frame)
        # accelerations = []
        # for i in range(0,len(velocity0)):
        #     accelerations.append(30.*(velocity1[i]-velocity1[i]))
        # return accelerations
        accelerations = self.getDOFAccelerations(frame)
        accelerationRoot = accelerations[0].copy()
        accelerations[0][3] = accelerationRoot[5] #Z
        accelerations[0][4] = accelerationRoot[3] #X
        accelerations[0][5] = accelerationRoot[4] #Y
        for i in range(1, len(accelerations)):
            acceleration = accelerations[i].copy()
            accelerations[i][0] = acceleration[2]
            accelerations[i][1] = acceleration[0]
            accelerations[i][2] = acceleration[1]
        return accelerations

    # [I<vmerge>R_g[0], R_l[1]^t, R_l[2]^t, ... R_l[n-1]^t]
    def getDOFAxeses(self, frame):
        # return [np.concatenate((mm.I_SO3(), mm.I_SO3()))] + [R.transpose() for R in self.getInternalJointOrientationsGlobal(frame)]
        return [np.concatenate((mm.I_SO3(), self[frame].getJointOrientationGlobal(0).transpose()))] + [R.transpose() for R in self.getInternalJointOrientationsGlobal(frame)]
    
    # [I<vmerge>R_g[0], R_l[1]^t, R_l[2]^t, ... R_l[n-1]^t]
    def getDOFAxesesLocal(self, frame):
        # return [np.concatenate((mm.I_SO3(), mm.I_SO3()))] + [R.transpose() for R in self.getInternalJointOrientationsGlobal(frame)]
        return [np.concatenate((mm.I_SO3(), self[frame].getJointOrientationGlobal(0).transpose()))] + [R.transpose() for R in self.getInternalJointOrientationsGlobal(frame)]
    
    def setDOFPositions(self, frame, DOFPositions, update=True):
        # R, p = mm.T2Rp(DOFPositions[0])
        # self[frame].rootPos = p
        # self.setJointOrientationsLocal(frame, [R]+DOFPositions[1:])
        self[frame].rootPos = DOFPositions[0][0]
        self.setJointOrientationsLocal(frame, [DOFPositions[0][1]]+DOFPositions[1:], update)
    
    def getJointOrientationLocal(self, jointIndex, frame):
        return self[frame].getJointOrientationLocal(jointIndex)

    def getJointAngVelocityLocal(self, jointIndex, frame0, frame1=None):
        return self._getDerivative(jointIndex, frame0, frame1, self.getJointOrientationLocal, lambda R1, R0: mm.logSO3(np.dot(R0.transpose(), R1)))

    def getJointAngAccelerationLocal(self, jointIndex, frame0, frame1=None):
        return self._getDerivative(jointIndex, frame0, frame1, self.getJointAngVelocityLocal, op.sub)

    def setJointOrientationLocal(self, jointIndex, frame, R, update=True):
        self[frame].setJointOrientationLocal(jointIndex, R, update)

    def getJointPositionGlobal(self, jointIndex, frame):
        return self[frame].getJointPositionGlobal(jointIndex)

    def getJointVelocityGlobal(self, jointIndex, frame0, frame1=None):
        return self._getDerivative(jointIndex, frame0, frame1, self.getJointPositionGlobal, op.sub)

    def getJointAccelerationGlobal(self, jointIndex, frame0, frame1=None):
        return self._getDerivative(jointIndex, frame0, frame1, self.getJointVelocityGlobal, op.sub)

    def getJointOrientationGlobal(self, jointIndex, frame):
        return self[frame].getJointOrientationGlobal(jointIndex)

    def getJointAngVelocityGlobal(self, jointIndex, frame0, frame1=None):
        return self._getDerivative(jointIndex, frame0, frame1, self.getJointOrientationGlobal, lambda R1, R0: np.dot(R0, mm.logSO3(np.dot(R0.transpose(), R1))))

    def getJointAngAccelerationGlobal(self, jointIndex, frame0, frame1=None):
        return self._getDerivative(jointIndex, frame0, frame1, self.getJointAngVelocityGlobal, op.sub)
    
    def setJointPositionGlobal(self, jointIndex, frame, position):
        return self[frame].setJointPositionGlobal(jointIndex, position)

    def setJointOrientationGlobal(self, jointIndex, frame, R, update=True):
        return self[frame].setJointOrientationGlobal(jointIndex, R, update)

    def getJointOrientationsLocal(self, frame):
        return self[frame].getJointOrientationsLocal()

    def getJointAngVelocitiesLocal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getJointOrientationsLocal, lambda R1, R0: mm.logSO3(np.dot(R0.transpose(), R1)))

    def getJointAngAccelerationsLocal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getJointAngVelocitiesLocal, op.sub)
    
    def setJointOrientationsLocal(self, frame, Rs, update=True):
        self[frame].setJointOrientationsLocal(Rs, update)
    
    def getJointPositionsGlobal(self, frame):
        return self[frame].getJointPositionsGlobal()

    def getJointVelocitiesGlobal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getJointPositionsGlobal, op.sub)

    def getJointAccelerationsGlobal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getJointVelocitiesGlobal, op.sub)

    def getJointOrientationsGlobal(self, frame):
        return self[frame].getJointOrientationsGlobal()

    def getJointAngVelocitiesGlobal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getJointOrientationsGlobal, lambda R1, R0: np.dot(R0, mm.logSO3(np.dot(R0.transpose(), R1))))

    def getJointAngAccelerationsGlobal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getJointAngVelocitiesGlobal, op.sub)
    
    def getInternalJointOrientationsLocal(self, frame):
        return self[frame].getInternalJointOrientationsLocal()

    def getInternalJointAngVelocitiesLocal(self, frame0, frame1=None):
#        # m2 - m1 : m2.__sub__(m1)
#        # d = m2 - m1
#        # R_d = inverse(R_m1) (dot) R_m2
#        orientations0 = self.getOrientationsLocal(frame0)
#        orientations1 = self.getOrientationsLocal(frame1)
#        return map(lambda R0,R1: (self.fps/(frame1-frame0)) * mm.logSO3(np.dot(R0.transpose(), R1)), orientations0, orientations1)
        return self._getDerivatives(frame0, frame1, self.getInternalJointOrientationsLocal, lambda R1, R0: mm.logSO3(np.dot(R0.transpose(), R1)))
    def getInternalJointAngAccelerationsLocal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getInternalJointAngVelocitiesLocal, op.sub)

    def getInternalJointPositionsGlobal(self, frame):
        return self[frame].getInternalJointPositionsGlobal()

    def getInternalJointVelocitiesGlobal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getInternalJointPositionsGlobal, op.sub)

    def getInternalJointAccelerationsGlobal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getInternalJointVelocitiesGlobal, op.sub)

    def getInternalJointOrientationsGlobal(self, frame):
        return self[frame].getInternalJointOrientationsGlobal()

    def getInternalJointAngVelocitiesGlobal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getInternalJointOrientationsGlobal, lambda R1, R0: np.dot(R0, mm.logSO3(np.dot(R0.transpose(), R1))))

    def getInternalJointAngAccelerationsGlobal(self, frame0, frame1=None):
        return self._getDerivatives(frame0, frame1, self.getInternalJointAngVelocitiesGlobal, op.sub)

    def rotateTrajectory(self, R_offset, fixedPos=None, update=True):
        if fixedPos is None:
            fixedPos = self[0].rootPos
        for p in self:
            relRootPos = p.rootPos - fixedPos
            relRootPos = np.dot(R_offset, relRootPos)
            p.rootPos = relRootPos + fixedPos
            p.rotateByOffset(R_offset, update)

    def rotateTrajectoryLocal(self, R_offset, fixedPos=None, update=True):
        if fixedPos==None: fixedPos = self[0].rootPos
        for p in self:
            relRootPos = p.rootPos - fixedPos
            relRootPos = np.dot(R_offset, relRootPos)
            p.rootPos = relRootPos + fixedPos
            p.rotateByOffsetLocal(R_offset, update)

    def updateGlobalT(self):
        for p in self:
            p.updateGlobalT()


class JointSkeleton(Skeleton):
    """
    :type root : Joint
    """
    def __init__(self, root):
        Skeleton.__init__(self)
        self.root = root
        self.rootIndex = None   # root always should be self.elements[0] and self.jointElementIndexes[0]
        self.jointElementIndexes = []
        self.reverseJointElementIndexes = {}

    def initialize(self):
        # build jointElementIndexes
        del self.jointElementIndexes[:]
        for i in range(self.getElementNum()):
            joint = self.elements[i]
            if len(joint.children) > 0:
                self.jointElementIndexes.append(i)
                self.reverseJointElementIndexes[i] = len(self.jointElementIndexes)-1

    def __str__(self):
        s = Skeleton.__str__(self)
        s += '<HIERARCHY>\n'
        s += self.root.__strHierarchy__()
        s += '<JOINT INDEX:(ELEMENT INDEX) ELEMENT NAME>\n'
        for i in range(len(self.jointElementIndexes)):
            s += '[%d]:(%d)\'%s\', '%(i, self.jointElementIndexes[i], self.elementNames[self.jointElementIndexes[i]])
        s += '\n'
        s += '<INTERNAL JOINT INDEX:(ELEMENT INDEX) ELEMENT NAME>\n'
        for i in range(1, len(self.jointElementIndexes)):
                s += '[%d]:(%d)\'%s\', '%(i-1, self.jointElementIndexes[i], self.elementNames[self.jointElementIndexes[i]])
        s += '\n'
        return s
    
    #===============================================================================
    # DOF functions
    #===============================================================================
    def getDOFs(self):
        return [6] + [3]*self.getInternalJointNum()

    def getTotalDOF(self):
        return 6 + 3*self.getInternalJointNum()

    #===============================================================================
    # joint index
    #===============================================================================
    def getJointNum(self):
        return len(self.jointElementIndexes)

    def getJoint(self, jointIndex):
        return self.elements[self.jointElementIndexes[jointIndex]] # type: Joint

    def getJointName(self, jointIndex):
        return self.elementNames[self.jointElementIndexes[jointIndex]]

    def getJointIndex(self, name):
        if name in self.reverseElementNames:
            index = self.reverseElementNames[name]
            return self.reverseJointElementIndexes[index]
        else:
            return None
    
    def getInternalJointNum(self):
        return len(self.jointElementIndexes)-1

    def getInternalJointName(self, internalJointIndex):
        return self.elementNames[self.jointElementIndexes[internalJointIndex+1]]
    
    def getLinkNum(self):
        return self.getJointNum()
    
    def getParentJointIndex(self, jointIndex):
        index = self.jointElementIndexes[jointIndex]
        parentIndex = self.getParentIndex(index)
        if parentIndex!=None:
            return self.reverseJointElementIndexes[parentIndex]
        else:
            return None
    
    #===============================================================================
    # element index
    #===============================================================================
    def getParentIndex(self, index):
        joint = self.elements[index]
        if joint.parent is not None:
            return self.reverseElementNames[joint.parent.name]
        else:
            return None

    def getChildIndexes(self, index):
        ls = []
        joint = self.elements[index]
        for child in joint.children:
            ls.append(self.getElementIndex(child.name))
        return ls

    def getOffset(self, index):
        return self.elements[index].offset

    def removeElement(self, index):
        # print(index)
        joint = Skeleton.getElement(self, index)
        parent = joint.parent

        for child in joint.children:
            child.parent = parent
            child.offset += joint.offset
        
        if parent is not None:
            # print([child.name for child in parent.children])
            # print([Skeleton.getElementIndex(self, child.name) for child in parent.children])
            index_in_parent = [Skeleton.getElementIndex(self, child.name) for child in parent.children].index(index)
            # print(index_in_parent)
            del parent.children[index_in_parent:index_in_parent+1]
            parent.children.extend(joint.children)
            # print([child.name for child in parent.children])
            
        Skeleton.removeElement(self, index)
        # print([child.name for child in parent.children])
        self.initialize()
        # print(self)
        # print([child.name for child in parent.children])

#    def changeRoot(self, root_name_or_index):
#        if isinstance(root_name_or_index, int):
#            self.root = self.elements[root_name_or_index]
#        else:
#            self.root = self.elements[self.getElementIndex(root_name_or_index)]
#        self._rebuildHierarchy(self.root, None)
#
#    def _rebuildHierarchy(self, node, parentNode):
#        linkedNodes = node.children + [node.parent]
#
#        node.parent = parentNode
#        if parentNode!=None:
#            parentNode.children.append(node)
#
#        del node.children[:]
#        for childCandidate in linkedNodes:
#            if childCandidate != parentNode and childCandidate != None:
#                self._rebuildHierarchy(childCandidate, node)

    def scale(self, scale):
        for i in range(self.getElementNum()):
            joint = self.getElement(i)
            joint.offset *= scale


class Joint:
    def __init__(self, name, parent):
        self.name = name  # type: str
        self.parent = parent  # type: Joint
        self.children = []  # type: list[Joint]
        self.offset = mm.O_Vec3()

    def addChild(self, name_or_joint, offset=None):
        if isinstance(name_or_joint, Joint):
            childJoint = name_or_joint
        else:
            name = name_or_joint
            childJoint = Joint(name, self)
        if offset!=None:
            childJoint.offset = offset
        self.children.append(childJoint)
        return childJoint

    def calcGlobalOffset(self):
        offset = mm.O_Vec3()
        current = self
        while current:
            offset += current.offset
            current = current.parent
        return offset

    def __str__(self):
        string = ''
        string += 'name: '+self.name
        string += '  offset: '+self.offset.__str__()
        # string += '  offset: '+repr(self.offset)
#        string += ', globalOffset: '+self.calcGlobalOffset().__str__()
        return string

    def __strHierarchy__(self, depth = 0):
        s = ''
        tab = '  '*depth
        s += '%s%s\n'%(tab, self.__str__())
        depth += 1
        for child in self.children:
            s += child.__strHierarchy__(depth)
        return s


class JointPosture(Posture):
    def __init__(self, skeleton):
        Posture.__init__(self, skeleton)
        self.skeleton = skeleton  # type: JointSkeleton
        self.local_ts = [None]*skeleton.getElementNum()
        self.localRs = [None]*skeleton.getElementNum()
        self.globalTs = [None]*skeleton.getElementNum()
        self.rootPos = mm.O_Vec3()

    # m1 + d : m1.__add__(d)
    # m2 = m1 + d
    # R_m2 = R_m1 (dot) R_d
    def __add__(self, displacement):
        """

        :param displacement: JointDisplacement
        :return:
        """
        if displacement.__class__ == JointDisplacement:
            m1 = self
            d = displacement
            m2 = JointPosture(self.skeleton)
            m2.rootPos = m1.rootPos + d.rootPos
            for i in range(len(m1.localRs)):
                R_m1 = m1.localRs[i]
                R_d = d.localRs[i]
                R_m2 = np.dot(R_m1, R_d) 
                m2.setLocalR(i, R_m2)
            m2.updateGlobalT()
            return m2
        else:
            raise TypeError

    def __sub__(self, other):
        # m2 - m1 : m2.__sub__(m1)
        # d = m2 - m1
        # R_d = inverse(R_m1) (dot) R_m2
        if other.__class__ == JointPosture:
            m2 = self
            m1 = other
            d = JointDisplacement(self.skeleton)
            d.rootPos = m2.rootPos - m1.rootPos
            for i in range(len(m2.localRs)):
                R_m2 = m2.localRs[i]
                R_m1 = m1.localRs[i]
                R_d = np.dot(R_m1.transpose(), R_m2) 
                d.setLocalR(i, R_d)
            return d
        # m2 - d : m2.__sub__(d)
        # m1 = m2 + (-d)
        # R_m1 = R_m2 (dot) inverse(R_d)
        elif other.__class__ == JointDisplacement:
            m2 = self 
            d = other
            m1 = m2 + (-d)
            return m1
        else:
            raise TypeError
    
#    def initialize(self):
#        self.rootPos = mm.O_Vec3()
#        self.initLocalRs()
#        self.initGlobalTs()
#    def initGlobalTs(self):
#        for i in range(self.skeleton.getElementNum()):
#            self.globalTs[i] = mm.I_SE3()
    def initLocalRs(self, initialRs=None):
        if initialRs is None:
            for i in range(self.skeleton.getElementNum()):
                self.localRs[i] = mm.I_SO3()
        else:
            self.localRs = initialRs

    def initLocal_ts(self, initial_ts=None):
        if initial_ts is None:
            for i in range(self.skeleton.getElementNum()):
                self.localRs[i] = mm.O_Vec3()
        else:
            self.localRs = initial_ts

        self.rootPos = self.local_ts[0]

    def getTPose(self, initialRs=None):
        tpose = JointPosture(self.skeleton)  # share skeleton of self
        tpose.rootPos = self.rootPos.copy()
        tpose.initLocalRs(initialRs)
        tpose.updateGlobalT()
        return tpose
    
    def copy(self):
        copied = JointPosture(self.skeleton)   # share skeleton of self
        copied.rootPos = self.rootPos.copy()
        for i in range(self.skeleton.getElementNum()):
            copied.localRs[i] = self.localRs[i].copy()
            if self.globalTs[i] is not None:
                copied.globalTs[i] = self.globalTs[i].copy()
        return copied
    
    def getLocalR(self, index):
        return self.localRs[index]

    def setLocalR(self, index, localR):
        self.localRs[index] = localR

    def mulLocalR(self, index, R):
        self.localRs[index] = np.dot(self.localRs[index], R)

    def setLocal_t(self, index, local_t):
        self.local_ts[index] = local_t

    def getLocalRFromParent(self, index):
        # Gp * Rl = Gi
        # Rl = Li (= Gp.transpose() * Gi)
        return self.localRs[index]

    def getGlobalRFromParent(self, index):
        # Rg * Gp = Gi
        # Rg = Gi * Gp.transpose()
        parent = self.skeleton.getParentIndex(index)
        if parent is None:
            Gp = mm.I_SO3()
        else:
            Gp = self.getGlobalR(parent)
        Gi = self.getGlobalR(index)
        Rg = np.dot(Gi, Gp.transpose()) 
        return Rg

    def getGlobalR(self, index):
        return mm.T2R(self.globalTs[index])
    
    # Gp : global rotation of parent of joint i 
    # Gi : global rotation of joint i
    # Gin : new global rotation of joint i
    # Li : local rotation of joint i
    # Lin : new local rotation of joint i
    def setGlobalR(self, index, globalR):
        # Gi = Gp * Li
        # Gin = Gp * Lin
        # Lin = Gp.transpose() * Gin
        Gin = globalR
        parent = self.skeleton.getParentIndex(index)
        if parent==None:
            Gp = mm.I_SO3()
        else:
            Gp = self.getGlobalR(parent)
        Lin = np.dot(Gp.transpose(), Gin)
        self.setLocalR(index, Lin)
        # self.updateGlobalT(index)

    def mulGlobalR(self, index, R):
        # Gi = Gp * Li
        # R * Gi = Gp * Lin
        # Lin = Gp.transpose() * R * Gi
        parent = self.skeleton.getParentIndex(index)
        if parent==None:
            Gp = mm.I_SO3()
        else:
            Gp = self.getGlobalR(parent)
        Gi = self.getGlobalR(index)
        Lin = np.dot(Gp.transpose(), np.dot(R, Gi))
        self.setLocalR(index, Lin)
        # self.updateGlobalT(index)
        
#    def calcGlobalPosition(self, index):
#        return mm.T2p(self._calcGlobalT(self.skeleton.getElement(index)))
#    def calcGlobalT(self, index):
#        return self._calcGlobalT(self.skeleton.getElement(index))
#    def _calcGlobalT(self, joint):
#        if joint.parent:
#            T = self._calcGlobalT(joint.parent)
#        else:
#            T = mm.p2T(self.rootPos)
#        T = np.dot(T, mm.p2T(joint.offset))
#
#        index = self.skeleton.getElementIndex(joint.name)
#        T = np.dot(T, mm.R2T(self.localRs[index]))
#        return T
    
    def getGlobalT(self, index):
        return self.globalTs[index]

#    def updateGlobalT(self, fromIndex=None):
#        if fromIndex==None:
#            self._updateGlobalT(self.skeleton.root, self.rootPos, mm.I_SO3())
#        else:
#            parent = self.skeleton.getParentIndex(fromIndex)
#            if parent==None:
#                self._updateGlobalT(self.skeleton.root, self.rootPos, mm.I_SO3())
#            else:
#                joint = self.skeleton.getElement(fromIndex)
#                R, p = mm.T2Rp(self.globalTs[parent])
#                self._updateGlobalT(joint, p, R)
#    def _updateGlobalT(self, joint, parent_p, parentR):
#        index = self.skeleton.getElementIndex(joint.name)
#        p = parent_p + np.dot(parentR, joint.offset)
#        R = np.dot(parentR, self.localRs[index])
#        self.globalTs[index] = mm.Rp2T(R, p)
#        for childJoint in joint.children:
#            self._updateGlobalT(childJoint, p, R)

    def updateGlobalT(self, fromIndex=None):
        if fromIndex is None:
            # self._updateGlobalT(self.skeleton.root, mm.p2T(self.rootPos))
            self._updateGlobalT(self.skeleton.root, mm.I_SE3())
        else:
            parent = self.skeleton.getParentIndex(fromIndex)
            if parent is None:
                # self._updateGlobalT(self.skeleton.root, mm.p2T(self.rootPos))
                self._updateGlobalT(self.skeleton.root, mm.I_SE3())
            else:
                joint = self.skeleton.getElement(fromIndex)
                self._updateGlobalT(joint, self.globalTs[parent])

    def _updateGlobalT(self, joint, parentT):
        if joint is self.skeleton.root:
            self.rootPos = self.local_ts[0]
        index = self.skeleton.getElementIndex(joint.name)
        T = np.dot(parentT, mm.p2T(joint.offset))
        T = np.dot(T, mm.p2T(self.local_ts[index]))
        T = np.dot(T, mm.R2T(self.localRs[index]))
        self.globalTs[index] = T
        for childJoint in joint.children:
            # if '_Effector' not in childJoint.name:
            #     self._updateGlobalT(childJoint, T)
            self._updateGlobalT(childJoint, T)

    # do not use!
    # have to replace to IK version
    def setPosition(self, index, position):
        prevPos = self.getPosition(index)
        offset = position - prevPos
        self.rootPos += offset
        self.updateGlobalT()
        
    def getPosition(self, index):
        return mm.T2p(self.globalTs[index])

    def getPositions(self):
        return [self.getPosition(i) for i in range(self.skeleton.getElementNum())]
    
    def blendPosture(self, otherPosture, t, update=True):
        p = JointPosture(self.skeleton)
        p.rootPos = mm.linearInterpol(self.rootPos, otherPosture.rootPos, t)
        for i in range(len(self.local_ts)):
            p.local_ts[i] = (1.-t) * self.local_ts[i] + t * otherPosture.local_ts[i]
        for i in range(len(self.localRs)):
            p.localRs[i] = mm.slerp(self.localRs[i], otherPosture.localRs[i], t)
            # p.localRs[i] = cm.slerp(self.localRs[i], otherPosture.localRs[i], t)
        if update:
            p.updateGlobalT()
        # p.updateGlobalT()
        return p

    # ===========================================================================
    # joint index based new functions
    # ===========================================================================
    def translateByOffset(self, p_offset, update=True):
        self.rootPos += p_offset
        if update:
            self.updateGlobalT()
        # self.updateGlobalT()
        
    def translateByTarget(self, p_target, targetJointIndex=0, update=True):
        p = self.getJointPositionGlobal(targetJointIndex)
        p_offset = p_target - p
        self.translateByOffset(p_offset, update)
        
    def rotateByOffset(self, R_offset, update=True):
        self.setJointOrientationGlobal(0, np.dot(R_offset, self.getJointOrientationGlobal(0)), update)
        # self.updateGlobalT()
        
    def rotateByTarget(self, R_target, targetJointIndex=0, update=True):
        # R_offset * R = R_target
        # R_offset = R_target * R.T
        # R_root_new = R_offset * R_root
        R = self.getJointOrientationGlobal(targetJointIndex)
        R_offset = np.dot(R_target, R.T)
        self.rotateByOffset(R_offset, update)

    def rotateByOffsetLocal(self, R_offset, update=True):
        self.setJointOrientationGlobal(0, np.dot(self.getJointOrientationGlobal(0), R_offset), update)

    def mulJointOrientationGlobal(self, jointIndex, R_offset, update=True):
        self.mulGlobalR(self.skeleton.jointElementIndexes[jointIndex], R_offset)
        if update:
            self.updateGlobalT(self.skeleton.jointElementIndexes[jointIndex])

    def mulJointOrientationLocal(self, jointIndex, R_offset, update=True):
        self.mulLocalR(self.skeleton.jointElementIndexes[jointIndex], R_offset)
        if update:
            self.updateGlobalT(self.skeleton.jointElementIndexes[jointIndex])

    def getJointPositionGlobal(self, jointIndex, localOffset=None):
        jointElementIndex = self.skeleton.jointElementIndexes[jointIndex]
        if localOffset is not None:
            return self.getPosition(jointElementIndex) + np.dot(self.getGlobalR(jointElementIndex), localOffset)
        return self.getPosition(jointElementIndex)

    def getJointOrientationLocal(self, jointIndex):
        return self.localRs[self.skeleton.jointElementIndexes[jointIndex]]

    def getJointOrientationGlobal(self, jointIndex):
        return self.getGlobalR(self.skeleton.jointElementIndexes[jointIndex])

    def getJointOrientationFromParentGlobal(self, jointIndex):
        return self.getGlobalRFromParent(self.skeleton.jointElementIndexes[jointIndex])

    def setJointPositionGlobal(self, jointIndex, position):
        raise NotImplementedError('use IK?')

    def setJointOrientationLocal(self, jointIndex, R, update=True):
        self.setLocalR(self.skeleton.jointElementIndexes[jointIndex], R)
        if update:
            self.updateGlobalT(self.skeleton.jointElementIndexes[jointIndex])
    def setJointOrientationGlobal(self, jointIndex, R, update=True):
        self.setGlobalR(self.skeleton.jointElementIndexes[jointIndex], R)
        if update:
            self.updateGlobalT(self.skeleton.jointElementIndexes[jointIndex])

    def getJointPositionsGlobal(self):
        return [self.getPosition(self.skeleton.jointElementIndexes[i]) for i in range(len(self.skeleton.jointElementIndexes))]

    def getJointOrientationsLocal(self):
        return [self.localRs[self.skeleton.jointElementIndexes[i]] for i in range(len(self.skeleton.jointElementIndexes))]

    def getJointOrientationsGlobal(self):
        return [self.getGlobalR(self.skeleton.jointElementIndexes[i]) for i in range(len(self.skeleton.jointElementIndexes))]
    
    def setJointOrientationsLocal(self, Rs, update=True):
        for i in range(len(self.skeleton.jointElementIndexes)):
            self.localRs[self.skeleton.jointElementIndexes[i]] = Rs[i]
        if update:
            self.updateGlobalT()

    def getInternalJointPositionsGlobal(self):
        return [self.getPosition(self.skeleton.jointElementIndexes[i]) for i in range(1, len(self.skeleton.jointElementIndexes))]

    def getInternalJointOrientationsLocal(self):
        return [self.localRs[self.skeleton.jointElementIndexes[i]] for i in range(1, len(self.skeleton.jointElementIndexes))]

    def getInternalJointOrientationsGlobal(self):
        return [self.getGlobalR(self.skeleton.jointElementIndexes[i]) for i in range(1, len(self.skeleton.jointElementIndexes))]

    #########################
    ## Additional
    def addJoint(self, skeleton, localR) :
        self.skeleton = skeleton
        self.localRs.append(localR)
        self.globalTs.append(localR)

    def getDOFPositions(self):
        return [(self.rootPos, self.getLocalR(0))] + self.getInternalJointOrientationsLocal()

    def getDOFAxeses(self):
        return [np.concatenate((mm.I_SO3(), self.getJointOrientationGlobal(0).transpose()))] + [R.transpose() for R in self.getInternalJointOrientationsGlobal()]

    def setDOFPositions(self, DOFPositions):
        self.rootPos = DOFPositions[0][0]
        self.setJointOrientationsLocal([DOFPositions[0][1]]+DOFPositions[1:])

    def get_q(self):
        q = np.zeros(self.skeleton.getTotalDOF())
        q[:3] = mm.logSO3(self.getLocalR(self.skeleton.jointElementIndexes[0]))
        q[3:6] = self.rootPos
        joint_num = self.skeleton.getJointNum()
        for i in range(1, joint_num):
            q[3*i+3:3*i+6] = mm.logSO3(self.getLocalR(self.skeleton.jointElementIndexes[i]))
        return q

    def set_q(self, q):
        self.setLocalR(self.skeleton.jointElementIndexes[0], mm.exp(q[:3]))
        self.rootPos[0] = q[3]
        self.rootPos[1] = q[4]
        self.rootPos[2] = q[5]
        for i in range(1, self.skeleton.getJointNum()):
            self.setLocalR(self.skeleton.jointElementIndexes[i], mm.exp(q[3*i+3:3*i+6]))

    def getJointFrame(self, jointIndex):
        return self.getGlobalT(self.skeleton.jointElementIndexes[jointIndex])

    def scale(self, scale, update=True):
        self.rootPos *= scale
        if update:
            self.updateGlobalT()


def dart_q_slerp(t, _from, _to):
    """

    :param t:
    :type t: float
    :param _from:
    :type _from: np.ndarray
    :param _to:
    :type _to: np.ndarray
    :return:
    """
    res = np.zeros_like(_from)

    joint_num = np.shape(_from)[0]//3

    for i in range(joint_num):
        index = range(3*i, 3*(i+1))
        if i == 1:
            res[index] = (1.-t) * _from[index] + t * _to[index]
        else:
            res[index] = mm.logSO3(mm.slerp(mm.exp(_from[index]), mm.exp(_to[index]), t))

    return res


class JointDisplacement(JointPosture):
    def __init__(self, jointSkeleton):
        JointPosture.__init__(self, jointSkeleton)
        self.initLocalRs()

    # -d : d.__neg__()
    # neg_d = -d
    # R_neg_d = inverse(R_d)
    def __neg__(self):
        d = self
        neg_d = JointDisplacement(self.skeleton)
        neg_d.rootPos = -d.rootPos
        for i in range(len(d.localRs)):
            R_d = d.localRs[i]
            R_neg_d = R_d.transpose()
            neg_d.setLocalR(i, R_neg_d)
        return neg_d

    def __add__(self, displacement):
        if isinstance(displacement, JointDisplacement):
            d1 = self
            d2 = displacement
            d3 = JointDisplacement(self.skeleton)
            d3.rootPos = d1.rootPos + d2.rootPos
            for i in range(len(d1.localRs)):
                R_d1 = d1.localRs[i]
                R_d2 = d2.localRs[i]
                R_d3 = np.dot(R_d1, R_d2)
                d3.setLocalR(i, R_d3)
            return d3
        else:
            raise TypeError

    def __sub__(self, displacement):
        if isinstance(displacement, JointDisplacement):
            return self + (-displacement)
        else:
            raise TypeError

    def __mul__(self, constant):
        d = JointDisplacement(self.skeleton)
        d.rootPos = self.rootPos * constant
        for i in range(len(self.localRs)):
            R = self.localRs[i]
            R_d = mm.exp(constant * mm.logSO3(R))
            # R_d = cm.exp(constant * cm.log(R))
            d.setLocalR(i, R_d)
        return d

    def __rmul__(self, constant):
        return self.__mul__(constant)

    def disableTranslation(self):
        self.rootPos = mm.O_Vec3()

    def disableRotations(self, jointIndexes):
        for jointIndex in jointIndexes:
            self.setJointOrientationLocal(jointIndex, mm.I_SO3(), False)

    def copy(self):
        copied = JointDisplacement(self.skeleton)   # share skeleton of self
        copied.rootPos = self.rootPos.copy()
        for i in range(self.skeleton.getElementNum()):
            copied.localRs[i] = self.localRs[i].copy()
        return copied


class MMPosture(Posture):
    def __init__(self, pointSkeleton = None):
        self.points = []
        self.pointMap = {}
        #        self.pointIdxMap = {}
        if pointSkeleton == None:
            self.skeleton = PointSkeleton()
        else:
            self.skeleton = pointSkeleton

    def addPoint(self, name, point):
        self.points.append(point)
        self.pointMap[name] = point
    #        self.pointIdxMap[name] = len(self.points)-1

    def addPoint2(self, point, name=None):
        if name is None:
            name = 'unnamed_%d'%len(self.points)
        self.points.append(point)
        self.pointMap[name] = point

    def getGlobalPos(self, partName_or_partIndex):
        if isinstance(partName_or_partIndex, int):
            return self.points[partName_or_partIndex]
        elif isinstance(partName_or_partIndex, str):
            return self.pointMap[partName_or_partIndex]
        else:
            raise TypeError

    def getPointPositions(self):
        return copy.deepcopy(self.points)


class MMSkeleton(Skeleton):
    def __init__(self):
        self.links = []
        self.pointSet = set()

    def addLink(self, name1, name2):
        self.links.append((name1, name2))
        self.pointSet.add(name1)
        self.pointSet.add(name2)

    def __str__(self):
        l = list(self.pointSet)
        l.sort()
        return l.__str__()

    def getPartNames(self):
        return list(self.pointSet)

    def getPart(self, partName):
        return partName


if __name__ == '__main__':
    from fltk import *
    import Resource.ysMotionLoader as yf
    import GUI.ysSimpleViewer_ori as ysv
    import Renderer.ysRenderer as yr
    import Motion.ysMotionConverter as ymc
    import Util.ysGlHelper as ygh
    import Util.ysPythonEx as ype
    import Motion.ysSkeletonEdit as yme


    def test_motions():
        pointMotion = yf.readTrcFile('../samples/Day7_Session2_Take01_-_walk.trc', .01)
        jointMotion = yf.readBvhFile('../samples/wd2_WalkSameSame00.bvh', .01)

        print('pointSkeleton')
        print(pointMotion[0].skeleton)
        print('jointSkeleton')
        print(jointMotion[0].skeleton)

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('pointMotion', yr.PointMotionRenderer(pointMotion, (0,255,0)))
        viewer.doc.addObject('pointMotion', pointMotion)
        viewer.doc.addRenderer('jointMotion', yr.JointMotionRenderer(jointMotion, (0,255,0)))
        viewer.doc.addObject('jointMotion', jointMotion)

        viewer.startTimer(1/pointMotion.fps)
        viewer.show()

        Fl.run()

    def test_Motion_getPosition_getPositions():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        jointMotion = yf.readBvhFile(bvhFilePath, .01)

        viewer = ysv.SimpleViewer()
        viewer.doc.addRenderer('woody2', yr.JointMotionRenderer(jointMotion, (0,0,255), yr.LINK_LINE))
        viewer.doc.addObject('woody2', jointMotion)

        rf = jointMotion[0].skeleton.getElementIndex('RightFoot')
        lf = jointMotion[0].skeleton.getElementIndex('LeftFoot')

        def extraDrawCallback():
            frame = viewer.getCurrentFrame()
            if frame < len(jointMotion)-1:
                for f in ype.frange(0., 1., .1):
                    ygh.drawPoint(jointMotion.getPosition(rf, frame + f), mm.Vec3(255,0,0)*f)
                    ygh.drawPoint(jointMotion.getPositions(frame + f)[lf], mm.Vec3(0,255,0)*f)

        viewer.setExtraDrawCallback(extraDrawCallback)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_setPosition():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        jointMotion = yf.readBvhFile(bvhFilePath, .01)

        viewer = ysv.SimpleViewer()
        viewer.doc.addRenderer('woody2', yr.JointMotionRenderer(jointMotion, (0,0,255), yr.LINK_LINE))
        viewer.doc.addObject('woody2', jointMotion)

        lf = jointMotion[0].skeleton.getElementIndex('LeftFoot')

        def preFrameCallback(frame):
            jointMotion[frame].setPosition(lf, (0,0,0))
            jointMotion[frame].updateGlobalT()

        viewer.setPreFrameCallback(preFrameCallback)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_getGlobalT():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        jointMotion = yf.readBvhFile(bvhFilePath, .01)

        viewer = ysv.SimpleViewer()
        viewer.doc.addRenderer('woody2', yr.JointMotionRenderer(jointMotion, (0,0,255), yr.LINK_LINE))
        viewer.doc.addObject('woody2', jointMotion)

        lul = jointMotion[0].skeleton.getElementIndex('LeftUpLeg')

        def preFrameCallback(frame):
            jointMotion[frame].mulLocalR(lul, mm.exp(mm.v3(0,0,1), math.pi/4.))
        #            jointMotion[frame].updateGlobalT(lul)

        def extraDrawCallback():
            frame = viewer.getCurrentFrame()
            for i in range(jointMotion[0].skeleton.getElementNum()):
                # ygh.drawPoint(jointMotion[frame].calcGlobalPosition(i), (255,0,0), 5.)
                ygh.drawPoint(jointMotion[frame].getPosition(i), (0,255,0), 5.)

        viewer.setPreFrameCallback(preFrameCallback)
        viewer.setExtraDrawCallback(extraDrawCallback)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_get_setslice():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        motion = yf.readBvhFile(bvhFilePath, .01)
        motion_getslice = motion[0:60]
        motion_setslice = copy.deepcopy(motion)

        # below both lines work well
        motion_setslice[0:0] = [motion[0]]*30
        #        motion_setslice[0:0] = motion

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('motion', yr.JointMotionRenderer(motion, (0,0,255)))
        viewer.doc.addObject('motion', motion)
        viewer.doc.addRenderer('motion_getslice', yr.JointMotionRenderer(motion_getslice, (255,0,0)))
        viewer.doc.addObject('motion_getslice', motion_getslice)
        viewer.doc.addRenderer('motion_setslice', yr.JointMotionRenderer(motion_setslice, (0,255,0)))
        viewer.doc.addObject('motion_setslice', motion_setslice)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_Motion_getVelocity_getVelocities():
        pointMotion = yf.readTrcFile('../samples/Day7_Session2_Take01_-_walk.trc', .01)
        jointMotion = yf.readBvhFile('../samples/wd2_WalkSameSame00.bvh', .01)

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('pointMotion', yr.PointMotionRenderer(pointMotion, (0,0,255)))
        viewer.doc.addObject('pointMotion', pointMotion)
        viewer.doc.addRenderer('jointMotion', yr.JointMotionRenderer(jointMotion, (0,0,255)))
        viewer.doc.addObject('jointMotion', jointMotion)

        #        motions = [pointMotion, jointMotion]
        motions = [jointMotion]
        def extraDrawCallback():
            frame = viewer.getCurrentFrame()

            for motion in motions:
                if frame < len(motion):
                    positions = motion[frame].getPositions()
                    velocities1 = motion.getVelocities(frame)
                    for i in range(len(positions)):
                        ygh.drawVector(velocities1[i], positions[i], (0,255,0))
                        ygh.drawVector(motion.getVelocity(i, frame), positions[i], (0,255,255))

                    if frame > 1 and frame < len(motion)-2:
                        frame0 = frame-2; frame1 = frame+2
                        velocities2 = motion.getVelocities(frame0, frame1)
                        for i in range(len(positions)):
                            ygh.drawVector(velocities2[i], positions[i], (255,0,0))
                            ygh.drawVector(motion.getVelocity(i, frame0, frame1), positions[i], (255,255,0))

        viewer.setExtraDrawCallback(extraDrawCallback)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_JointDisplacement():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        jointMotion = yf.readBvhFile(bvhFilePath, .01)

        print(isinstance(JointDisplacement(jointMotion[0].skeleton), JointPosture))
        print(JointDisplacement(jointMotion[0].skeleton).__class__ == JointPosture)

        m1 = jointMotion[50:51]
        m2 = jointMotion[70:71]
        d = Motion([m2[0] - m1[0]])

        m2_1 = Motion([m1[0] + d[0]])

        m2_2 = copy.deepcopy(m1)
        m2_2[0] += d[0]

        m1_1 = Motion([m2[0] - d[0]])

        m1_2 = copy.deepcopy(m2)
        m1_2[0] -= d[0]

        d_map = []
        for i in range(10):
            k = i*.1
            d_map.append(k*d[0])


        viewer = ysv.SimpleViewer()
        viewer.record(False)

        viewer.doc.addRenderer('jointMotion', yr.JointMotionRenderer(jointMotion, (0,0,255)))
        viewer.doc.addObject('jointMotion', jointMotion)
        viewer.doc.addRenderer('m1', yr.JointMotionRenderer(m1, (0,0,255)))
        viewer.doc.addRenderer('m2', yr.JointMotionRenderer(m2, (255,0,0)))
        viewer.doc.addRenderer('d', yr.JointMotionRenderer(d, (0,255,0)))
        viewer.doc.addRenderer('m2_1', yr.JointMotionRenderer(m2_1, (0,255,0)))
        viewer.doc.addRenderer('m2_2', yr.JointMotionRenderer(m2_2, (255,255,0)))
        viewer.doc.addRenderer('m1_1', yr.JointMotionRenderer(m1_1, (0,255,255)))
        viewer.doc.addRenderer('m1_2', yr.JointMotionRenderer(m1_2, (255,255,255)))
        for i in range(10):
            viewer.doc.addRenderer('%d'%i, yr.JointMotionRenderer(Motion([m1[0]+d_map[i]]), (255,i*25,255)))

        viewer.startTimer(1/30.)
        viewer.show()
        Fl.run()

    def test_blendPosture():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        jointMotion = yf.readBvhFile(bvhFilePath, .01)

        posture0 = jointMotion[0]
        posture1 = jointMotion[50]
        interpolatedPostures = Motion()

        steps = 10
        for i in range(steps):
            interpolatedPostures.append(posture0.blendPosture(posture1, float(i)/steps))

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('motion', yr.JointMotionRenderer(jointMotion, (0,255,0), yr.LINK_LINE))
        viewer.doc.addObject('motion', jointMotion)

        viewer.doc.addRenderer('posture0', yr.JointMotionRenderer(Motion([posture0]), (0,0,255), yr.LINK_LINE))
        viewer.doc.addObject('posture0', jointMotion)

        viewer.doc.addRenderer('posture1', yr.JointMotionRenderer(Motion([posture1]), (255,0,0), yr.LINK_LINE))
        viewer.doc.addObject('posture1', jointMotion)

        ipr = yr.JointMotionRenderer(interpolatedPostures, (255,255,0), yr.LINK_LINE)
        ipr.renderFrames = [i for i in range(len(interpolatedPostures))]
        viewer.doc.addRenderer('interpolatedPostures', ipr)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_element_pos_get_acc_funcs():
        #        motion = yf.readBvhFile('../samples/wd2_WalkSameSame00.bvh', .01)
        #        motion = yf.readBvhFile('../samples/block_3_rotate.bvh', 1)
        motion = yf.readBvhFile('../samples/block_tree_rotate.bvh', 1)
        print(motion[0].skeleton)

        p = []
        v = []

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('motion', yr.JointMotionRenderer(motion, (0,0,255), yr.LINK_WIREBOX))
        viewer.doc.addObject('motion', motion)
        viewer.doc.addRenderer('v', yr.VectorsRenderer(v, p, (0,255,0)))

        def preFrameCallback(frame):
            p[:] = motion.getPositions(frame)
            v[:] = motion.getVelocities(frame)

        viewer.setPreFrameCallback(preFrameCallback)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_joint_pos_vel_acc_funcs():
        #        bvhFilePath = '../samples/block_tree_rotate.bvh'
        #        bvhFilePath = '../samples/chain_3_rotate.bvh'
        bvhFilePath = '../samples/chain_2_rotate_2axes.bvh'
        motion = yf.readBvhFile(bvhFilePath)
        print(motion[0].skeleton)

        #        yme.rotateJointLocal(motion, 0, mm.exp(mm.v3(0,1,0), math.pi/2))

        p = []
        ap = []; av = []; aa = []

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('motion', yr.JointMotionRenderer(motion, (0,0,255), yr.LINK_WIREBOX))
        viewer.doc.addObject('motion', motion)
        #        viewer.doc.addRenderer('ap', yr.VectorsRenderer(ap, p, (255,0,0)))
        viewer.doc.addRenderer('av', yr.VectorsRenderer(av, p, (255,255,0)))
        #        viewer.doc.addRenderer('aa', yr.VectorsRenderer(aa, p, (0,255,255)))

        def preFrameCallback(frame):
            p[:] = motion.getJointPositionsGlobal(frame)

            #            ap[:] = map(mm.logSO3, motion.getJointOrientationsGlobal(frame))
            av[:] = motion.getJointAngVelocitiesGlobal(frame)
            #            av[:] = motion.getJointAngVelocitiesLocal(frame)
            aa[:] = motion.getJointAngAccelerationsGlobal(frame)

        viewer.setPreFrameCallback(preFrameCallback)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_translate_rotate_scale():
        pm = yf.readTrcFile('../samples/Day7_Session2_Take01_-_walk.trc', .01)
        jm = yf.readBvhFile('../samples/wd2_WalkSameSame00.bvh', .01)
        print('pointSkeleton')
        print(pm[0].skeleton)
        print('jointSkeleton')
        print(jm[0].skeleton)

        pm_translateByOffset = copy.deepcopy(pm)
        pm_translateByOffset.translateByOffset((0,0,.2))

        jm_translateByOffset = copy.deepcopy(jm)
        jm_translateByOffset.translateByOffset((0,0,.2))

        jm_translateByTarget = copy.deepcopy(jm)
        for i in range(len(jm)):
            jm_translateByTarget[i].translateByTarget(jm[i].getJointPositionGlobal(10), 14)

        jm_rotateByTarget = copy.deepcopy(jm)
        for i in range(len(jm)):
            jm_rotateByTarget[i].rotateByTarget(mm.I_SO3(), 1)

        jm_rotateTrajectory = copy.deepcopy(jm)
        jm_rotateTrajectory.rotateTrajectory(mm.rotY(math.pi/2.))

        jm_scale = copy.deepcopy(jm)
        jm_scale.scale(.5)

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        #        viewer.doc.addRenderer('pm', yr.PointMotionRenderer(pm, (0,100,255)))
        #        viewer.doc.addObject('pm', pm)
        #        viewer.doc.addRenderer('pm_translateByOffset', yr.PointMotionRenderer(pm_translateByOffset, (0,255,0)))
        #        viewer.doc.addObject('pm_translateByOffset', pm_translateByOffset)
        viewer.doc.addRenderer('jm', yr.JointMotionRenderer(jm, (0,100,255)))
        viewer.doc.addObject('jm', jm)
        #        viewer.doc.addRenderer('jm_translateByOffset', yr.JointMotionRenderer(jm_translateByOffset, (0,255,0)))
        #        viewer.doc.addObject('jm_translateByOffset', jm_translateByOffset)
        #        viewer.doc.addRenderer('jm_translateByTarget', yr.JointMotionRenderer(jm_translateByTarget, (255,255,0)))
        #        viewer.doc.addObject('jm_translateByTarget', jm_translateByTarget)
        #        viewer.doc.addRenderer('jm_rotateByTarget', yr.JointMotionRenderer(jm_rotateByTarget, (255,0,0)))
        #        viewer.doc.addObject('jm_rotateByTarget', jm_rotateByTarget)
        viewer.doc.addRenderer('jm_rotateTrajectory', yr.JointMotionRenderer(jm_rotateTrajectory, (0,255,255)))
        viewer.doc.addObject('jm_rotateTrajectory', jm_rotateTrajectory)
        #        viewer.doc.addRenderer('jm_scale', yr.JointMotionRenderer(jm_scale, (255,0,255)))
        #        viewer.doc.addObject('jm_scale', jm_scale)

        viewer.startTimer(1/30.)
        viewer.show()

        Fl.run()

    def test_JointMotion_setDOFPositions():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        jointMotion = yf.readBvhFile(bvhFilePath, .01)
        jointMotion2 = copy.deepcopy(jointMotion)

        positions = [mm.I_SO3()]*jointMotion2[0].skeleton.getJointNum()
        positions.insert(0, (mm.O_Vec3(), mm.I_SO3()))

        jointMotion2.setDOFPositions(0, positions)

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('jointMotion', yr.JointMotionRenderer(jointMotion, (0,0,255)))
        viewer.doc.addObject('jointMotion', jointMotion)
        viewer.doc.addRenderer('jointMotion2', yr.JointMotionRenderer(jointMotion2, (0,255,0)))
        viewer.doc.addObject('jointMotion2', jointMotion2)

        viewer.startTimer(1/30.)
        viewer.show()
        Fl.run()

        pass


    pass
    #    test_motions()
    #    test_JointDisplacement()
    #    test_Motion_getVelocity_getVelocities()
    test_get_setslice()
    # test_getGlobalT()
    #    test_Motion_getPosition_getPositions()
    #    test_setPosition()
    #    test_blendPosture()
    #    test_element_pos_get_acc_funcs()
    #    test_joint_pos_vel_acc_funcs()
    #    test_translate_rotate_scale()
    #    test_JointMotion_setDOFPositions()
