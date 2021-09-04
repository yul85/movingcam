import numpy
import math

from PyCommon.modules.Math import mmMath as mm
# from PyCommon.modules.Math import csMath as cm
from PyCommon.modules.Motion import ysMotion as ym
from PyCommon.modules.Motion import pmMotion as pm

import copy

ROOT_NAME = "root"


# ===============================================================================
# .mm file
# ===============================================================================
def readMMFile(mmFilePath):
    names = {}
    names[0] = 'root'
    names[2] = 'lKnee'
    names[4] = 'neck'
    names[6] = 'rKnee'
    names[8] = 'rFoot'
    names[10] = 'rToe'
    names[12] = 'head'
    names[14] = 'rArm'
    names[16] = 'lArm'
    names[18] = 'lHand'
    names[20] = 'rHand'
    #    names[14] = 'lArm'
    #    names[16] = 'rArm'
    #    names[18] = 'rHand'
    #    names[20] = 'lHand'
    names[22] = 'lFoot'
    names[24] = 'lToe'

    skeleton = ym.MMSkeleton()
    skeleton.addLink("head", "neck")
    skeleton.addLink("neck", "lArm")
    skeleton.addLink("lArm", "lHand")
    skeleton.addLink("neck", "rArm")
    skeleton.addLink("rArm", "rHand")
    skeleton.addLink("neck", "root")
    skeleton.addLink("root", "lKnee")
    skeleton.addLink("lKnee", "lFoot")
    skeleton.addLink("lFoot", "lToe")
    skeleton.addLink("root", "rKnee")
    skeleton.addLink("rKnee", "rFoot")
    skeleton.addLink("rFoot", "rToe")

    #    # lowest foot height finding code
    #    lowest = 100
    #    for mmFilePath in paths:
    #        pointMotion = yf.readMMFile(mmFilePath)
    #        i = 0
    #        for name, point in pointMotion[i].pointMap.items():
    #            if name == 'rFoot' or name == 'lFoot' or name == 'rToe' or name == 'lToe':
    #                if point[1] < lowest:
    #                    print mmFilePath, i
    #                    print name
    #                    print point[1]
    #                    lowest = point[1]
    lowest = .15

    f = open(mmFilePath)
    fileLines = f.readlines()
    pointMotion = ym.Motion()
    i = 0
    while i != len(fileLines):
        if fileLines[i].isspace() == False:
            splited = fileLines[i].split()
            pointPosture = ym.MMPosture(skeleton)
            for j in range(0, len(splited), 2):
                point = numpy.array([float(splited[j]), float(splited[j + 1]), 0.])
                point[1] -= lowest
                pointPosture.addPoint(names[j], point)
            pointMotion.append(pointPosture)
        i += 1

    f.close()
    pointMotion.fps = 30.
    return pointMotion


# ===============================================================================
# .trc file
# ===============================================================================
def readTrcFile(trcFilePath, scale=1.0):
    f = open(trcFilePath)
    fileLines = f.readlines()
    pointMotion = ym.Motion()
    i = 0
    while i != len(fileLines):
        splited = fileLines[i].split()
        boneNames = []
        if i == 2:
            dataRate = float(splited[0])
            numFrames = int(splited[2])
            numMarkers = int(splited[3])
        #            print numFrames, numMarkers
        elif i == 3:
            markerNames = [name.split(':')[1] for name in splited[2:]]
            skeleton = ym.PointSkeleton()
            for name in markerNames:
                skeleton.addElement(None, name)
        #            print markerNames
        elif i > 5:
            markerPositions = splited[2:]
            #            print markerPositions
            #            print 'i', i
            pointPosture = ym.PointPosture(skeleton)
            for m in range(numMarkers):
                point = numpy.array([float(markerPositions[m * 3]), float(markerPositions[m * 3 + 1]),
                                     float(markerPositions[m * 3 + 2])])
                point = numpy.dot(mm.exp(numpy.array([1, 0, 0]), -math.pi / 2.), point) * scale
                #                pointPosture.addPoint(markerNames[m], point)
                pointPosture.setPosition(m, point)
            #                print 'm', m
            #                print markerNames[m], (markerPositions[m*3],markerPositions[m*3+1],markerPositions[m*3+2])
            pointMotion.append(pointPosture)
        i += 1
    f.close()
    pointMotion.fps = dataRate
    return pointMotion


# ===============================================================================
# .bvh file
# ===============================================================================
def readBvhFile(bvhFilePath, scale=1.0, applyRootOffset=False):
    bvh = Bvh()
    bvh.parseBvhFile(bvhFilePath)
    jointMotion = bvh.toJointMotion(scale, applyRootOffset)
    return jointMotion


def readBvhFileAsBvh(bvhFilePath):
    bvh = Bvh()
    bvh.parseBvhFile(bvhFilePath)
    return bvh


def writeBvhFile(bvhFilePath, jointMotion):
    bvh = Bvh()
    bvh.fromJointMotion(jointMotion)
    bvh.writeBvhFile(bvhFilePath)


class Bvh:
    channelTypes6dof = ['XPOSITION', 'YPOSITION', 'ZPOSITION', 'ZROTATION', 'XROTATION', 'YROTATION']
    channelTypes3dof = ['ZROTATION', 'XROTATION', 'YROTATION']

    class Joint:
        def __init__(self, name):
            self.name = name
            self.offset = None  # type: numpy.array
            self.channels = []  # type: list[Bvh.Channel]
            self.children = []  # type: list[Bvh]
            self.jointIndex = None  # type: int

        def __strHierarchy__(self, depth=0):
            s = ''
            tab1 = '  ' * depth
            tab2 = '  ' * (depth + 1)
            s += '%sJOINT %s\n' % (tab1, self.name)
            s += '%s{\n' % tab1
            s += '%sOFFSET %s\n' % (tab2, self.offset)

            channelString = ''
            for channel in self.channels:
                channelString += channel.__str__() + ' '
            s += '%sCHANNELS %s\n' % (tab2, channelString)

            for child in self.children:
                s += child.__strHierarchy__(depth + 1)
            s += '%s}\n' % tab1
            return s

    class Channel:
        def __init__(self, channelType, channelIndex):
            self.channelType = channelType  # type: str
            self.channelIndex = channelIndex  # type: int

        def __str__(self):
            return self.channelType

    def __init__(self):
        self.joints = []  # type: list[Bvh.Joint]
        self.frameNum = 0
        self.frameTime = 0
        self.motionList = []

        self.totalChannelCount = 0

    def __str__(self):
        s = 'HIERARCHY\n'
        s += self.joints[0].__strHierarchy__()
        s += 'MOTION\n'
        s += 'Frame: %d\n' % self.frameNum
        s += 'Frame Time: %f\n' % self.frameTime
        #        for i in range(len(self.motionList)):
        #            s += self.motionList[i].__str__() + '\n'
        return s

    # ===========================================================================
    # read functions
    # ===========================================================================
    def parseBvhFile(self, filepath_or_fileobject):
        if isinstance(filepath_or_fileobject, str):
            f = open(filepath_or_fileobject)
        else:
            f = filepath_or_fileobject

        tokens = f.read().split()
        tokens.reverse()

        self.totalChannelCount = 0
        self.parseBvhHierachy(tokens)
        self.parseBvhMotion(tokens)

    def parseBvhHierachy(self, tokens):
        if tokens.pop().upper() != "HIERARCHY":
            print("HIERARCHY missing")
            return
        if tokens.pop().upper() != "ROOT":
            print("ROOT missing")
            return
        self.parseBvhJoint(tokens.pop(), tokens)

    def parseBvhJoint(self, name, tokens):
        bvhJoint = Bvh.Joint(name)
        self.joints.append(bvhJoint)

        if tokens.pop() != "{":
            print("'{' missing")
            return None

        endDetected = False
        while not endDetected:
            t = tokens.pop().upper()
            if t == '}':
                endDetected = True
            elif t == 'OFFSET':
                x = float(tokens.pop())
                y = float(tokens.pop())
                z = float(tokens.pop())
                bvhJoint.offset = numpy.array([x, y, z], float)
            elif t == 'CHANNELS':
                channelCount = int(tokens.pop())
                for i in range(channelCount):
                    channelType = tokens.pop().upper()
                    bvhJoint.channels.append(Bvh.Channel(channelType, self.totalChannelCount))
                    self.totalChannelCount += 1
            elif t == 'JOINT':
                bvhJoint.children.append(self.parseBvhJoint(tokens.pop(), tokens))
            elif t == 'END':
                next = tokens.pop().upper()
                if next != 'SITE':
                    print('END', next, 'is unknown keyword')
                bvhJoint.children.append(self.parseBvhJoint("%s_Effector" % name, tokens))
            else:
                print("invalid bvhJoint definition")
                return None
        return bvhJoint

    def parseBvhMotion(self, tokens):
        if tokens.pop().upper() != 'MOTION':
            print("MOTION missing")
            return None
        if tokens.pop().upper() != 'FRAMES:':
            print("FRAMES: missing")
            return None
        self.frameNum = int(tokens.pop())
        if tokens.pop().upper() != 'FRAME TIME:':
            if tokens.pop().upper() != 'TIME:':
                print("FRAME TIME: missing")
                return None
        self.frameTime = float(tokens.pop())

        self.motionList = [None] * self.frameNum
        for i in range(self.frameNum):
            self.motionList[i] = [None] * self.totalChannelCount

        for i in range(self.frameNum):
            for j in range(self.totalChannelCount):
                self.motionList[i][j] = float(tokens.pop())

    # ===========================================================================
    # write functions
    # ===========================================================================
    def writeBvhFile(self, filepath_or_fileobject):
        if isinstance(filepath_or_fileobject, str):
            f = open(filepath_or_fileobject, 'w')
        else:
            f = filepath_or_fileobject

        self.writeBvhHierarchy(f)
        self.writeBvhMotion(f)
        print(self)

    def writeBvhHierarchy(self, file):
        file.write('HIERARCHY\n')
        self.writeBvhJoint(file, self.joints[0], 0)

    def writeBvhJoint(self, file, joint, depth):
        indent_joint = '  ' * depth
        indent_offset = '  ' * (depth + 1)

        if len(joint.children) > 0:
            endsite = False
        else:
            endsite = True

        # JOINT
        if not endsite:
            if depth == 0:
                joint_label = 'ROOT'
            else:
                joint_label = 'JOINT'
            file.write('%s%s %s\n' % (indent_joint, joint_label, joint.name))
        else:
            file.write('%sEnd Site\n' % indent_joint)
        file.write('%s{\n' % indent_joint)

        # OFFSET
        file.write('%sOFFSET %s %s %s\n' % (indent_offset, joint.offset[0], joint.offset[1], joint.offset[2]))

        if not endsite:
            # CHANNELS
            channelString = ''
            for channel in joint.channels:
                channelString += channel.__str__() + ' '
            file.write('%sCHANNELS %d %s\n' % (indent_offset, len(joint.channels), channelString))

            # children
            for child in joint.children:
                self.writeBvhJoint(file, child, depth + 1)

        # end JOINT
        file.write('%s}\n' % indent_joint)

    def writeBvhMotion(self, file):
        file.write('MOTION\n')
        file.write('Frames: %d\n' % self.frameNum)
        file.write('Frame Time: %f\n' % self.frameTime)

        for i in range(self.frameNum):
            for j in range(self.totalChannelCount):
                file.write('%s ' % self.motionList[i][j])
            file.write('\n')

    # ===========================================================================
    # modification functions
    # ===========================================================================
    def set_scale(self, scale=1.0):
        # set joint offset scale
        for joint in self.joints:
            joint.offset *= scale

        # set channel scale
        for joint in self.joints:
            for channel in joint.channels:
                if 'POSITION' in channel.channelType:
                    for k in range(self.frameNum):
                        self.motionList[k][channel.channelIndex] *= scale

    # ===========================================================================
    # conversion functions
    # ===========================================================================
    def toJointMotion(self, scale, applyRootOffset):
        skeleton = self.toJointSkeleton(scale, applyRootOffset)

        jointMotion = ym.JointMotion()
        for i in range(len(self.motionList)):
            jointPosture = ym.JointPosture(skeleton)
            self.addJointSO3FromBvhJoint(jointPosture, self.joints[0], self.motionList[i], scale)
            jointPosture.updateGlobalT()
            jointMotion.append(jointPosture)

        jointMotion.fps = 1. / self.frameTime
        return jointMotion

    def toJointSkeleton(self, scale, applyRootOffset):
        # build joint hierarchy
        jointMap = {}
        root = self.addJointFromBvhJoint(jointMap, self.joints[0].name, self.joints[0], None, scale, applyRootOffset)

        # build joint array
        skeleton = ym.JointSkeleton(root)
        for bvhJoint in self.joints:
            skeleton.addElement(jointMap[bvhJoint.name], bvhJoint.name)
        skeleton.rootIndex = skeleton.getElementIndex(root.name)

        # initialize
        skeleton.initialize()

        return skeleton

    def toPmLinearMotion(self, scale, applyRootOffset):
        skeleton = self.toPmHuman(scale, applyRootOffset)

        pm_motion = pm.PmLinearMotion()
        for i in range(len(self.motionList)):
            pm_posture = pm.PmPosture(skeleton)
            self.addJointSO3FromBvhJoint(pm_posture, self.joints[0], self.motionList[i], scale)
            pm_posture.updateGlobalT()
            pm_motion.append(pm_posture)

        pm_motion.fps = 1. / self.frameTime
        return pm_motion

    def toPmHuman(self, scale, applyRootOffset):
        # build joint hierarchy
        jointMap = {}
        root = self.addJointFromBvhJoint(jointMap, self.joints[0].name, self.joints[0], None, scale, applyRootOffset)

        # build joint array
        skeleton = pm.PmHuman(root)
        for bvhJoint in self.joints:
            skeleton.addElement(jointMap[bvhJoint.name], bvhJoint.name)
        skeleton.rootIndex = skeleton.getElementIndex(root.name)

        # initialize
        skeleton.initialize()

        return skeleton

    def addJointFromBvhJoint(self, jointMap, jointName, bvhJoint, parentJoint, scale, applyOffset):
        joint = ym.Joint(jointName, parentJoint)
        if applyOffset:
            joint.offset = bvhJoint.offset * scale
        jointMap[jointName] = joint

        for i in range(len(bvhJoint.children)):
            child = self.addJointFromBvhJoint(jointMap, bvhJoint.children[i].name, bvhJoint.children[i], joint, scale,
                                              True)
            joint.children.append(child)

        return joint

    def addJointSO3FromBvhJoint(self, jointPosture, bvhJoint, channelValues, scale=1.0):
        localR = mm.I_SO3()
        local_t = mm.O_Vec3()

        for channel in bvhJoint.channels:
            if channel.channelType == 'XPOSITION':
                # jointPosture.rootPos[0] = channelValues[channel.channelIndex]*scale
                local_t[0] = channelValues[channel.channelIndex] * scale
            elif channel.channelType == 'YPOSITION':
                # jointPosture.rootPos[1] = channelValues[channel.channelIndex]*scale
                local_t[1] = channelValues[channel.channelIndex] * scale
            elif channel.channelType == 'ZPOSITION':
                # jointPosture.rootPos[2] = channelValues[channel.channelIndex]*scale
                local_t[2] = channelValues[channel.channelIndex] * scale
            elif channel.channelType == 'XROTATION':
                localR = numpy.dot(localR, mm.exp(mm.s2v((1, 0, 0)), mm.deg2Rad(channelValues[channel.channelIndex])))
            elif channel.channelType == 'YROTATION':
                localR = numpy.dot(localR, mm.exp(mm.s2v((0, 1, 0)), mm.deg2Rad(channelValues[channel.channelIndex])))
            elif channel.channelType == 'ZROTATION':
                localR = numpy.dot(localR, mm.exp(mm.s2v((0, 0, 1)), mm.deg2Rad(channelValues[channel.channelIndex])))
        # jointPosture.setLocalR(bvhJoint.name, localR)
        jointPosture.setLocalR(jointPosture.skeleton.getElementIndex(bvhJoint.name), localR)
        jointPosture.setLocal_t(jointPosture.skeleton.getElementIndex(bvhJoint.name), local_t)

        for i in range(len(bvhJoint.children)):
            self.addJointSO3FromBvhJoint(jointPosture, bvhJoint.children[i], channelValues, scale)

    # ===========================================================================
    # JointMotion -> Bvh
    # ===========================================================================
    def fromJointMotion(self, jointMotion):
        skeleton = jointMotion[0].skeleton
        self.fromJointSkeleton(skeleton)

        self.frameNum = len(jointMotion)
        self.motionList = [[None] * self.totalChannelCount for i in range(self.frameNum)]
        for i in range(self.frameNum):
            self._jointValue2channelValues(jointMotion[i], self.motionList[i], skeleton, self.joints[0])

        self.frameTime = 1. / jointMotion.fps

    def fromJointSkeleton(self, jointSkeleton):
        # build bvh joint hierarchy
        bvhJointDict = {}
        self.totalChannelCount = 0
        bvhRoot = self._Joint2BvhJoint(jointSkeleton.getElement(0), bvhJointDict)

        # build bvh joint array
        self.joints = [None] * jointSkeleton.getElementNum()
        for i in range(jointSkeleton.getElementNum()):
            self.joints[i] = bvhJointDict[jointSkeleton.getElementName(i)]

    def _Joint2BvhJoint(self, joint, bvhJointDict):
        bvhJoint = Bvh.Joint(joint.name)  # name
        bvhJointDict[joint.name] = bvhJoint

        bvhJoint.offset = joint.offset  # offset

        # channels
        if joint.parent is None:
            channelTypes = Bvh.channelTypes6dof
        elif len(joint.children) == 0:
            channelTypes = []
        else:
            channelTypes = Bvh.channelTypes3dof

        for channelType in channelTypes:
            bvhJoint.channels.append(Bvh.Channel(channelType, self.totalChannelCount))
            self.totalChannelCount += 1

        # children
        for child in joint.children:
            bvhJoint.children.append(self._Joint2BvhJoint(child, bvhJointDict))

        return bvhJoint

    # jointPosture : input
    # channelValues : output
    def _jointValue2channelValues(self, jointPosture, channelValues, jointSkeleton, bvhJoint):
        jointIndex = jointSkeleton.getElementIndex(bvhJoint.name)

        #todo : rotation to euler angle_zxy

        def R2ZXY(rm):

            r00 = rm[0]
            r01 = rm[1]
            r02 = rm[2]
            r10 = rm[3]
            r11 = rm[4]
            r12 = rm[5]
            r20 = rm[6]
            r21 = rm[7]
            r22 = rm[8]

            if r21 < 1:
                if r21 > -1:
                    x = math.asin(r21)
                    z = math.atan2(-r01, r11)
                    y = math.atan2(-r20, r22)
                else:
                    x = -math.pi/2
                    z = math.atan2(r02, r00)
                    y = 0
            else:
                x = math.pi/2
                z = math.atan2(r02, r00)
                y = 0

            return z, x, y

        # zrot, xrot, yrot = cm.R2zxy_r(jointPosture.getLocalR(jointIndex))
        zrot, xrot, yrot = R2ZXY(jointPosture.getLocalR(jointIndex))

        for channel in bvhJoint.channels:
            if channel.channelType == 'XPOSITION':
                channelValues[channel.channelIndex] = jointPosture.rootPos[0]
            elif channel.channelType == 'YPOSITION':
                channelValues[channel.channelIndex] = jointPosture.rootPos[1]
            elif channel.channelType == 'ZPOSITION':
                channelValues[channel.channelIndex] = jointPosture.rootPos[2]
            elif channel.channelType == 'XROTATION':
                channelValues[channel.channelIndex] = xrot * mm.DEG
            elif channel.channelType == 'YROTATION':
                channelValues[channel.channelIndex] = yrot * mm.DEG
            elif channel.channelType == 'ZROTATION':
                channelValues[channel.channelIndex] = zrot * mm.DEG

        for child in bvhJoint.children:
            self._jointValue2channelValues(jointPosture, channelValues, jointSkeleton, child)

    # ===========================================================================
    # replace joint
    # ===========================================================================
    # planeString : 'XY', 'YZ', 'ZX'
    def mirror(self, planeString):
        offsetIdx = 3
        if planeString == 'YZ':
            offsetIdx = 0
        elif planeString == 'ZX':
            offsetIdx = 1
        elif planeString == 'XY':
            offsetIdx = 2
        else:
            print('Specify valid plane')
            return
        for joint in self.joints:
            joint.offset[offsetIdx] = -joint.offset[offsetIdx]

    def findJointDescendentIdxs(self, _joint):
        idxs = []
        for joint in _joint.children:
            jointIdx = self.joints.index(joint)
            idxs.append(jointIdx)
            idxs.extend(self.findJointDescendentIdxs(joint))

        return idxs

    def getJointFromJointName(self, jointName):
        joint = None
        for _joint in self.joints:
            if _joint.name == jointName:
                joint = _joint
                break

        return joint

    def replaceJointFromBvh(self, jointName, partBvh, scale=1.0):
        # find a partroot joint in original bvh
        rootJoint = self.getJointFromJointName(jointName)
        rootJointIdx = self.joints.index(rootJoint)

        if rootJoint is None:
            print('No Joint named ', jointName)
            return

        # ------------------------------------------
        # delete
        # ------------------------------------------
        # remove decendant joints of a partroot joint
        def multi_delete(list_, args):
            args.sort(reverse=True)
            for index in args:
                del list_[index]
            return list_

        # remove motion of removed joints
        removeJointList = self.findJointDescendentIdxs(rootJoint)
        removeMotionList = []
        removeMotionIdx = 0
        for jointIdx in range(len(self.joints)):
            jointChannelNum = len(self.joints[jointIdx].channels)
            if jointIdx in removeJointList:
                for j in range(jointChannelNum):
                    removeMotionList.append(removeMotionIdx + j)
            removeMotionIdx += jointChannelNum

        self.joints = multi_delete(self.joints, removeJointList)

        for _motion in self.motionList:
            multi_delete(_motion, removeMotionList)

        # modify joint channel indices
        for joint in self.joints:
            for channel in joint.channels:
                channel.channelIndex -= sum([k < channel.channelIndex for k in removeMotionList])

        # modify total channel count in bvh
        self.totalChannelCount = 0
        for joint in self.joints:
            self.totalChannelCount += len(joint.channels)

        # ------------------------------------------
        # add
        # ------------------------------------------
        # modify joint name
        for joint in partBvh.joints[1:]:
            joint.name = rootJoint.name + '_' + joint.name

        # modify channel idx
        partChannelCount = partBvh.totalChannelCount - len(partBvh.joints[0].channels)
        rootJointChannelIdx = rootJoint.channels[-1].channelIndex
        for joint in self.joints:
            for channel in joint.channels:
                channel.channelIndex += partChannelCount if rootJointChannelIdx < channel.channelIndex else 0

        # attach part bvh joints in partroot joint and bvh
        rootJoint.children = []
        rootJoint.children.extend(partBvh.joints[0].children)
        for jointIdx in range(1, len(partBvh.joints[1:]) + 1):
            self.joints.insert(rootJointIdx + jointIdx, partBvh.joints[jointIdx])

        # modify total channel count in bvh
        self.totalChannelCount = 0
        for joint in self.joints:
            self.totalChannelCount += len(joint.channels)

        # add new motion for part bvh
        for motionFrame in range(len(self.motionList)):
            for i in range(partChannelCount):
                self.motionList[motionFrame].insert(rootJointChannelIdx + 1, .0)

        # modify channel idx remained
        partJointChannelIdx = 0
        for joint in partBvh.joints[1:]:
            for channel in joint.channels:
                partJointChannelIdx += 1
                channel.channelIndex = rootJointChannelIdx + partJointChannelIdx

        # modify joint index
        for joint in self.joints:
            joint.jointIndex = self.joints.index(joint)

        # modify joint offset
        for joint in partBvh.joints:
            joint.offset *= scale

    # ===========================================================================
    # other motion file type import
    # ===========================================================================
    def importFromAsfAmc(self, asf_path, amc_path):
        self.importAsf(asf_path)
        self.importAmc(amc_path)

    def importAsf(self, asf_path):
        bvhJoint = Bvh.Joint(name)
        pass

    def importAmc(self, amc_path):
        pass


if __name__ == "__main__":
    import time
    import cProfile, os
    from datetime import datetime
    from fltk import *
    import GUI.ysSimpleViewer as ysv
    import Renderer.ysRenderer as yr

    import Simulator.csVpModel as cvm
    import Simulator.csVpWorld as cvw
    import cPickle
    import Simulator.ysPhysConfig as ypc
    import Renderer.csVpRenderer as cvr


    def test_readTrcFile():
        trcMotion = readTrcFile('../samples/Day7_Session2_Take01_-_walk.trc', .01)
        print(trcMotion[0].skeleton)

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('trcMotion', yr.PointMotionRenderer(trcMotion, (0, 255, 0)))
        viewer.doc.addObject('trcMotion', trcMotion)

        viewer.startTimer(1 / trcMotion.fps)
        viewer.show()

        Fl.run()


    def test_parseBvhFile():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        bvh = Bvh()
        bvh.parseBvhFile(bvhFilePath)
        print(bvh)


    def test_readBvhFile():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        motion = readBvhFile(bvhFilePath, .01)
        motion2 = readBvhFile(bvhFilePath, .01, True)
        print(motion[0].skeleton)

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('motion', yr.JointMotionRenderer(motion, (0, 255, 0)))
        viewer.doc.addObject('motion', motion)
        viewer.doc.addRenderer('motion2', yr.JointMotionRenderer(motion2, (255, 0, 0)))
        viewer.doc.addObject('motion2', motion2)

        viewer.startTimer(1 / motion.fps)
        viewer.show()

        Fl.run()


    def test_writeBvhFile():
        #        # bvh
        #        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        #        bvh = readBvhFileAsBvh(bvhFilePath)
        #
        #        tempFilePath = '../samples/bvh_wd2_WalkSameSame00.bvh.temp'
        #        bvh.writeBvhFile(tempFilePath)

        # motion
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        motion = readBvhFile(bvhFilePath, .01)
        #        motion[0] = motion[0].getTPose()

        tempFilePath = '../samples/motion_temp_wd2_WalkSameSame00.bvh.temp'
        writeBvhFile(tempFilePath, motion)
        motion2 = readBvhFile(tempFilePath)
        #        motion2[0] = motion2[0].getTPose()

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('motion', yr.JointMotionRenderer(motion, (0, 255, 0)))
        viewer.doc.addObject('motion', motion)
        viewer.doc.addRenderer('motion2', yr.JointMotionRenderer(motion2, (255, 0, 0)))
        viewer.doc.addObject('motion2', motion2)

        viewer.startTimer(1 / motion.fps)
        viewer.show()

        Fl.run()


    def profile_readBvhFile():
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'

        profileDataFile = '../samples/cProfile_%s.profile' % datetime.today().strftime('%y%m%d_%H%M%S')
        cProfile.runctx('motion = readBvhFile(bvhFilePath)', globals(), locals(), profileDataFile)
        os.system('python ../../Tools/pprofui.py %s' % profileDataFile)


    def test_replaceBvhFile():
        # motion
        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        bvh = readBvhFileAsBvh(bvhFilePath)
        motion = bvh.toJointMotion(.01, False)

        partBvhFilePath = '../samples/simpleJump_long.bvh'
        partBvh = readBvhFileAsBvh(partBvhFilePath)
        bvh.replaceJointFromBvh('RightFoot', partBvh)

        partBvh = readBvhFileAsBvh(partBvhFilePath)
        partBvh.mirror('YZ')
        bvh.replaceJointFromBvh('LeftFoot', partBvh)

        motion2 = bvh.toJointMotion(.01, False)

        def buildMassMap():
            massMap = {}
            massMap = massMap.fromkeys(['Head', 'Head_Effector', 'Hips',
                                        'LeftArm', 'LeftFoot', 'LeftForeArm', 'LeftHand', 'LeftHand_Effector',
                                        'LeftLeg', 'LeftShoulder1', 'LeftUpLeg',
                                        'RightArm', 'RightFoot', 'RightForeArm', 'RightHand', 'RightHand_Effector',
                                        'RightLeg', 'RightShoulder', 'RightUpLeg',
                                        'Spine', 'Spine1',
                                        'RightFoot_foot_0_0', 'RightFoot_foot_0_1', 'RightFoot_foot_0_1_Effector',
                                        'RightFoot_foot_1_0', 'RightFoot_foot_1_1', 'RightFoot_foot_1_1_Effector',
                                        'RightFoot_foot_2_0', 'RightFoot_foot_2_1', 'RightFoot_foot_2_1_Effector',
                                        'LeftFoot_foot_0_0', 'LeftFoot_foot_0_1', 'LeftFoot_foot_0_1_Effector',
                                        'LeftFoot_foot_1_0', 'LeftFoot_foot_1_1', 'LeftFoot_foot_1_1_Effector',
                                        'LeftFoot_foot_2_0', 'LeftFoot_foot_2_1', 'LeftFoot_foot_2_1_Effector',
                                        ], 0.)

            # torso : 10
            massMap['Hips'] += 2.
            massMap['Spine'] += 8.

            # head : 3
            massMap['Spine1'] += 3.

            # right upper arm : 2
            massMap['RightArm'] += 2.

            # left upper arm : 2
            massMap['LeftArm'] += 2.

            # right lower arm : 1
            massMap['RightForeArm'] = 1.
            #    massMap['RightForeArm'] = 2.

            # left lower arm : 1
            massMap['LeftForeArm'] = 1.
            #    massMap['LeftForeArm'] = 2.

            # right thigh : 7
            massMap['Hips'] += 2.
            massMap['RightUpLeg'] += 5.

            # left thigh : 7
            massMap['Hips'] += 2.
            massMap['LeftUpLeg'] += 5.

            # right shin : 5
            massMap['RightLeg'] += 5.

            # left shin : 5
            massMap['LeftLeg'] += 5.

            # right foot : 4
            massMap['RightFoot'] += 2.

            # left foot : 4
            massMap['LeftFoot'] += 2.
            massMap['RightFoot_foot_0_0'] = .3
            massMap['RightFoot_foot_0_1'] = .3
            massMap['RightFoot_foot_1_0'] = .3
            massMap['RightFoot_foot_1_1'] = .3
            massMap['RightFoot_foot_2_0'] = .3
            massMap['RightFoot_foot_2_1'] = .3
            massMap['LeftFoot_foot_0_0'] = .3
            massMap['LeftFoot_foot_0_1'] = .3
            massMap['LeftFoot_foot_1_0'] = .3
            massMap['LeftFoot_foot_1_1'] = .3
            massMap['LeftFoot_foot_2_0'] = .3
            massMap['LeftFoot_foot_2_1'] = .3

            return massMap

        def buildMcfg():
            massMap = buildMassMap()
            mcfg = ypc.ModelConfig()
            mcfg.defaultDensity = 1000.
            mcfg.defaultBoneRatio = .9

            totalMass = 0.
            for name in massMap:
                node = mcfg.addNode(name)
                node.mass = massMap[name]
                totalMass += node.mass

            node = mcfg.getNode('Hips')
            node.length = .2
            node.width = .25

            node = mcfg.getNode('Spine1')
            node.length = .2
            node.offset = (0, 0, 0.1)

            node = mcfg.getNode('Spine')
            node.width = .22

            node = mcfg.getNode('RightFoot')
            node.length = .25
            #    node.length = .27
            #    node.offset = (0,0,0.01)
            node.width = .1
            node.geom = 'MyFoot1'

            node = mcfg.getNode('LeftFoot')
            node.length = .25
            #    node.length = .27
            #    node.offset = (0,0,0.01)
            node.width = .1
            node.geom = 'MyFoot1'

            def capsulize(node_name):
                node = mcfg.getNode(node_name)
                node.geom = 'MyFoot4'
                node.width = 0.02

            capsulize('RightFoot')
            capsulize('LeftFoot')
            capsulize('RightFoot_foot_0_0')
            capsulize('RightFoot_foot_0_1')
            capsulize('RightFoot_foot_1_0')
            capsulize('RightFoot_foot_1_1')
            capsulize('RightFoot_foot_2_0')
            capsulize('RightFoot_foot_2_1')
            capsulize('LeftFoot_foot_0_0')
            capsulize('LeftFoot_foot_0_1')
            capsulize('LeftFoot_foot_1_0')
            capsulize('LeftFoot_foot_1_1')
            capsulize('LeftFoot_foot_2_0')
            capsulize('LeftFoot_foot_2_1')

            return mcfg

        mcfg = buildMcfg()

        wcfg = ypc.WorldConfig()
        wcfg.planeHeight = 0.
        wcfg.useDefaultContactModel = False
        wcfg.lockingVel = 0.05
        stepsPerFrame = 30
        wcfg.timeStep = 0.001

        print(motion2[0].skeleton)

        vpWorld = cvw.VpWorld(wcfg)
        motionModel = cvm.VpMotionModel(vpWorld, motion2[0], mcfg)
        # vpWorld.initialize()

        viewer = ysv.SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('motion', yr.JointMotionRenderer(motion, (0, 255, 0)))
        viewer.doc.addObject('motion', motion)
        viewer.doc.addRenderer('motion2', yr.JointMotionRenderer(motion2, (255, 0, 0)))
        viewer.doc.addObject('motion2', motion2)
        viewer.doc.addRenderer('csmotion2', cvr.VpModelRenderer(motionModel, (255, 0, 0), yr.POLYGON_FILL))

        viewer.startTimer(1 / motion.fps)

        def callback(frame):
            motionModel.update(motion2[frame])

        viewer.setSimulateCallback(callback)

        viewer.show()

        Fl.run()

        tempFilePath = '../samples/motion_foot_temp_wd2_WalkSameSame00.bvh.temp'
        # writeBvhFile(tempFilePath, motion2)


    pass
    #    test_readTrcFile()
    #    test_parseBvhFile()
    #    test_readBvhFile()
    #     test_writeBvhFile()
    #    profile_readBvhFile()
    test_replaceBvhFile()