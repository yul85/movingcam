from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLE import *
# import ode, numpy
import numpy
import math
import copy

import sys
if '..' not in sys.path:
    sys.path.append('..')
from PyCommon.modules.Math import mmMath as mm
from PyCommon.modules.Util import ysGlHelper as ygh
from PyCommon.modules.Motion import ysMotion as ym
# from PyCommon.modules.Mesh import ysMesh as yms
# import Math.mmMath as mm
# import Util.ysGlHelper as ygh
# import Motion.ysMotion as ym
# import Mesh.ysMesh as yms

# from PyCommon.modules.Renderer import csRenderer as cr
# from PyCommon.modules.Renderer.csRenderer import ObjImporter


# for hinting
# from PyCommon.modules.pyVirtualPhysics import *
# from PyCommon.modules.Simulator import csVpUtil as cvu
# from PyCommon.modules.Simulator import csDartModel as cdm
import pydart2 as pydart
import fltk

# RendererContext
NORMAL_FLAT = 0
NORMAL_SMOOTH = 1

# RendererContext, OdeModelRenderer
POLYGON_LINE = 0
POLYGON_FILL = 1

# JointMotionRenderer
LINK_LINE = 0
LINK_BONE = 1
LINK_SOLIDBOX = 2
LINK_WIREBOX = 3

# RenderContext. PointsRenderer
POINT_POINT = 0
POINT_CROSS = 1
POINT_CUBE = 2 

# SELECTION_COLOR = (10,10,.7)
SELECTION_COLOR = (2550,2550,178)

RENDER_OBJECT = 0
RENDER_SHADOW = 1
RENDER_REFLECTION = 2

shadow_color = (50, 50, 50)

FOOT_RENDER_ONLY = False
LEFT_FOOT_ONLY = False
RIGHT_FOOT_ONLY = False

CAPSULE_SLICE_SIZE = 8
SPHERE_SLICE_SIZE = 64


class Renderer:
    """

    :type rc: RenderContext
    :type totalColor: tuple | list | numpy.ndarray
    :type savedState: list | None
    """
    def __init__(self, target, color):
        self.rc = RenderContext()
        self.totalColor = color
        self.selectedElement = None
        self.shadowColor = (150,150,150)
        self.savedState = list()
        self.savable = False

    def render(self, renderType=RENDER_OBJECT):
        print("Renderer.render() : Must subclass me")
        raise NotImplementedError

    def renderState(self, state, renderType=RENDER_OBJECT):
        self.render(renderType)

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        self.render(renderType)

    def getState(self):
        return None

    def saveState(self):
        self.savedState.append(self.getState())

    def get_max_saved_frame(self):
        return len(self.savedState) - 1

    def resetSavedState(self):
        if self.savable:
            del self.savedState[:]


class SelectedGeomRenderer(Renderer):
    def __init__(self, color):
        Renderer.__init__(self, None, color = (255,0,0))
        self.geom = None
        self.rc.setPolygonStyle(POLYGON_LINE)
    def render(self, renderType=RENDER_OBJECT):
        if self.geom:
            glColor3ubv(self.totalColor)
            self.rc.renderSelectedOdeGeom(self.geom, self.totalColor)


class OdeRenderer(Renderer):
    def __init__(self, target, color = (255,255,255)):
        Renderer.__init__(self, target, color)
        self.space = target
    def render(self, renderType=RENDER_OBJECT):
        glColor3ubv(self.totalColor)
        for i in range(self.space.getNumGeoms()):
            geom = self.space.getGeom(i)

            if geom == self.selectedElement:
                glColor3ubv(SELECTION_COLOR)

            self.rc.renderOdeGeom(geom)

            if geom == self.selectedElement:
                glColor3ubv(self.totalColor)

class OdeModelRenderer(Renderer):
    def __init__(self, target, color = (255,255,255), polygonStyle = POLYGON_FILL):
        Renderer.__init__(self, target, color)
        self.model = target
        self.rc.setPolygonStyle(polygonStyle)
    def render(self, renderType=RENDER_OBJECT):
        glColor3ubv(self.totalColor)
        for node in self.model.nodes.values():
            geom = node.geom
            #            if node.name in self.partColors:
            #                glColor3ubv(self.partColors[node.name])
            #            else:
            #                glColor3ubv(self.totalColor)
            if geom == self.selectedElement:
                glColor3ubv(SELECTION_COLOR)

            self.rc.renderOdeGeom(geom)

            if geom == self.selectedElement:
                glColor3ubv(self.totalColor)

'''
class VpPyModelRenderer(Renderer):
    """
    # :type model : csVpModel_py.VpModel
    """
    def __init__(self, target, color=(255,255,255), polygonStyle=POLYGON_FILL, lineWidth=1.):
        Renderer.__init__(self, target, color)
        self.model = target
        self.rc.setPolygonStyle(polygonStyle)
        self._lineWidth = lineWidth

    def render(self, renderType=RENDER_OBJECT):
        glLineWidth(self._lineWidth)

        if renderType == RENDER_SHADOW:
            glColor3ub(90, 90, 90)
        else:
            glColor3ubv(self.totalColor)

        for node in self.model._nodes:
            if node is not None:
                if (renderType != RENDER_SHADOW):
                    if node.color[0] != 0 or node.color[1] != 0 or node.color[2] != 0:
                        c = [ node.color[0], node.color[1], node.color[2], node.color[3] ]
                        glColor4ubv(c)
                    else:
                        glColor3ubv(self.totalColor)
                self.renderVpNode(node)

        # if renderType!=RENDER_SHADOW:
        #     glDisable(GL_BLEND)

    def renderVpNode(self, pNode):
        glPushMatrix()
        _T = pNode.body.GetFrame()

        glMultMatrixd(cvu.SE3_2_pySE3(_T).T)

        for j in range(len(pNode.geoms)):
            pGeom = pNode.geoms[j]
            glPushMatrix()
            # _T = SE3_2_pySE3(pGeom.GetLocalFrame())
            _T = pGeom.GetLocalFrame()

            glMultMatrixd(cvu.SE3_2_pySE3(_T).T)

            geomType = pGeom.GetType()
            data = []
            if geomType ==  'B' or geomType == 'M':
                data = pGeom.GetSize()
                glPushMatrix()
                glTranslatef(-data[0]/2., -data[1]/2., -data[2]/2.)
                self.rc.drawBox(data[0], data[1], data[2])
                glPopMatrix()
            elif geomType == 'C':
                data.append(pGeom.GetRadius())
                data.append(pGeom.GetHeight())
                data[1] -= 2. * data[0]
                # self.rc.drawCylinder(data[0], data[1])
                self.rc.drawCapsule(data[0], data[1])
            elif geomType == 'S':
                data.append(pGeom.GetRadius())
                self.rc.drawSphere(data[0])
            glPopMatrix()
        glPopMatrix()

    def renderState(self, state, renderType=RENDER_OBJECT):
        """

        :type state: list[tuple[str, numpy.ndarray, numpy.ndarray]]
        :return:
        """
        glLineWidth(self._lineWidth)

        for elem in state:
            geomType, geomT, data, color = elem
            glPushMatrix()
            glMultMatrixd(geomT.transpose())
            if renderType != RENDER_SHADOW:
                glColor3ubv(color)
            else:
                glColor3ub(90, 90, 90)

            if geomType == 'B' or geomType == 'M':
                glTranslatef(-data[0]/2., -data[1]/2., -data[2]/2.)
                self.rc.drawBox(data[0], data[1], data[2])
            elif geomType == 'C':
                self.rc.drawCapsule(data[0], data[1])
            elif geomType == 'S':
                self.rc.drawSphere(data[0])

            glPopMatrix()

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        self.renderState(self.savedState[frame], renderType)

    def getState(self):
        state = []
        for node in self.model._nodes:
            color = None
            if node is not None:
                if node.color[0] != 0 or node.color[1] != 0 or node.color[2] != 0:
                    c = ( node.color[0], node.color[1], node.color[2])
                    color = copy.deepcopy(c)
                else:
                    color = copy.deepcopy(self.totalColor)
                bodyFrame = cvu.SE3_2_pySE3(node.body.GetFrame())
                for geom in node.geoms:
                    geomT = numpy.dot(bodyFrame, cvu.SE3_2_pySE3(geom.GetLocalFrame()))
                    geomType = geom.GetType()
                    data = []
                    if geomType ==  'B' or geomType == 'M':
                        geomSize = geom.GetSize()
                        data = [geomSize[i] for i in range(3)]
                    elif geomType == 'C':
                        data.append(geom.GetRadius())
                        data.append(geom.GetHeight())
                        data[1] -= 2. * data[0]
                    elif geomType == 'S':
                        data.append(geom.GetRadius())
                    state.append((geomType, geomT, data, color))
        return state
'''


class VpWorldRenderer(Renderer):
    def __init__(self, target, color, polygonStyle=POLYGON_FILL, lineWidth=1.):
        Renderer.__init__(self, target, color)
        self._world = target
        self._color = color
        self._polygonStyle = polygonStyle
        self._lineWidth = lineWidth
        self.rc.setPolygonStyle(polygonStyle)
        self.savable = True

    def render(self, renderType=RENDER_OBJECT):
        if self._polygonStyle == POLYGON_FILL:
            glPolygonMode(GL_FRONT, GL_FILL)
        else:
            glPolygonMode(GL_FRONT, GL_LINE)
        glLineWidth(self._lineWidth)

        # sphere_bump_list = self._world.get_sphere_bump_list()

        for sphere_bump in self._world.get_sphere_bump_list():
            if renderType == RENDER_OBJECT:
                glColor3ubv(self._color)
                glPushMatrix()
                glTranslatef(sphere_bump[1][0], sphere_bump[1][1], sphere_bump[1][2])
                self.rc.drawSphere(sphere_bump[0])
                glPopMatrix()

        if False:
            for plane in self._world.get_plane_list():
                if renderType == RENDER_OBJECT:
                    plane_normal = plane[0]
                    plane_origin = plane[1]
                    box_center = plane_origin - plane_normal*0.05
                    glColor3ubv(self._color)
                    glPushMatrix()
                    glTranslatef(box_center[0], box_center[1], box_center[2])
                    rot_vec = mm.logSO3(mm.getSO3FromVectors(mm.unitY(), plane_normal))
                    angle = numpy.linalg.norm(rot_vec)
                    if angle > 0.00001:
                        axis = rot_vec/angle
                        glRotatef(mm.rad2Deg(angle), axis[0], axis[1], axis[2])
                    self.rc.drawCenteredBox(4., 0.1, 4.)
                    glPopMatrix()


class VpModelRenderer(Renderer):
    def __init__(self, target, color, polygonStyle=POLYGON_FILL, lineWidth=1.):
        Renderer.__init__(self, target, color)
        self._model = target
        self._color = color
        self._polygonStyle = polygonStyle
        self._lineWidth = lineWidth
        self.rc.setPolygonStyle(polygonStyle)
        self.body_colors = [color] * self._model.getBodyNum()
        self.geom_colors = [None] * self._model.getBodyNum()
        self.savable = True

    def render(self, renderType=RENDER_OBJECT):
        if self._polygonStyle == POLYGON_FILL:
            glPolygonMode(GL_FRONT, GL_FILL)
        else:
            glPolygonMode(GL_FRONT, GL_LINE)
        glLineWidth(self._lineWidth)

        # if renderType != RENDER_SHADOW:
        #     glEnable(GL_BLEND)
        #     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for i in range(self._model.getBodyNum()):
            if renderType == RENDER_SHADOW:
                # glColor3ub(90, 90, 90)
                glColor3ubv(shadow_color)
            else:
                glColor3ubv(self.body_colors[i])
            self.renderVpBody(i)

        if renderType != RENDER_SHADOW:
            glDisable(GL_BLEND)

    def renderVpBody(self, body_idx):
        # print(body_idx, self._model.index2name(body_idx), self._model.getBodyShape(body_idx))
        # print(self._model.index2name(body_idx), self._model.getBodyGeomsType(body_idx), self._model.getBodyGeomsSize(body_idx))
        # print(self._model.index2name(body_idx), self._model.getBodyGeomsGlobalFrame(body_idx))

        geom_types = self._model.getBodyGeomsType(body_idx)
        geom_sizes = self._model.getBodyGeomsSize(body_idx)
        geom_frames = self._model.getBodyGeomsGlobalFrame(body_idx)

        for i in range(self._model.getBodyGeomNum(body_idx)):
            geom_type, _T, geom_size = geom_types[i], geom_frames[i], geom_sizes[i]

            glPushMatrix()
            glMultMatrixd(_T.T)

            if geom_type in ('B', 'M', 'N'):
                data = geom_size
                glTranslated(-.5*data[0], -.5*data[1], -.5*data[2])
                self.rc.drawBox(data[0], data[1], data[2])
            elif geom_type in ('C', 'D', 'E'):
                data = geom_size
                # data.append(pGeom.GetRadius())
                # data.append(pGeom.GetHeight())
                data[1] -= 2. * data[0]
                self.rc.drawCapsule(data[0], data[1])
            elif geom_type == 'S':
                data = geom_size
                self.rc.drawSphere(data[0])

            glPopMatrix()

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1:
            self.renderState(self.getState(), renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def getState(self):
        state = []
        for body_idx in range(self._model.getBodyNum()):
            geom_types = self._model.getBodyGeomsType(body_idx)
            geom_sizes = self._model.getBodyGeomsSize(body_idx)
            geom_frames = self._model.getBodyGeomsGlobalFrame(body_idx)
            # geom_colors = self._color
            geom_colors = [self.body_colors[body_idx]]*self._model.getBodyGeomNum(body_idx) if self.geom_colors[body_idx] is None else self.geom_colors[body_idx]

            for i in range(self._model.getBodyGeomNum(body_idx)):
                state.append((body_idx, geom_types[i], geom_frames[i], geom_sizes[i], geom_colors[i]))
        return state

    def renderState(self, state, renderType=RENDER_OBJECT):
        if self._polygonStyle == POLYGON_FILL:
            glPolygonMode(GL_FRONT, GL_FILL)
        else:
            glPolygonMode(GL_FRONT, GL_LINE)
        for elem in state:
            body_idx, geom_type, _T, geom_size, color = elem

            if renderType == RENDER_OBJECT:
                glColor3ubv(color)
            elif renderType == RENDER_SHADOW:
                # glColor3ub(90, 90, 90)
                glColor3ubv(shadow_color)

            glPushMatrix()
            glMultMatrixd(_T.T)

            if geom_type in ('B', 'M', 'N'):
                # box case
                data = geom_size
                glTranslated(-.5*data[0], -.5*data[1], -.5*data[2])
                if not FOOT_RENDER_ONLY:
                    self.rc.drawBox(data[0], data[1], data[2])
            elif geom_type in ('C', 'D', 'E', 'F'):
                # capsule case
                data = geom_size.copy()
                # data.append(pGeom.GetRadius())
                # data.append(pGeom.GetHeight())
                data[1] -= 2. * data[0]
                body_name = self._model.index2name(body_idx)
                if LEFT_FOOT_ONLY:
                    if 'L' in body_name:
                        self.rc.drawCapsule(data[0], data[1])
                elif RIGHT_FOOT_ONLY:
                    if 'R' in body_name:
                        self.rc.drawCapsule(data[0], data[1])
                else:
                    self.rc.drawCapsule(data[0], data[1])
            elif geom_type == 'S':
                data = geom_size
                self.rc.drawSphere(data[0])

            glPopMatrix()


class DartRenderer(Renderer):
    """
    :type world: pydart.World
    """
    def __init__(self, target, color=(255, 255, 255), polygonStyle=POLYGON_FILL, lineWidth=1., save_state=True):
        Renderer.__init__(self, target, color)
        self.world = target
        self.rc.setPolygonStyle(polygonStyle)
        self._lineWidth = lineWidth
        self.savable = save_state

    def render(self, renderType=RENDER_OBJECT):
        if True:
            self.renderState(self.getState(), renderType)
        else:
            glLineWidth(self._lineWidth)

            if renderType == RENDER_SHADOW:
                # glColor3ubv(shadow_color)
                glColor3ubv(self.totalColor)
            else:
                glColor3ubv(self.totalColor)

            for skeleton in self.world.skeletons:
                for body in skeleton.bodynodes:
                    glPushMatrix()
                    glMultMatrixd(body.world_transform().transpose())
                    for shapeNode in body.shapenodes:
                        if shapeNode.has_visual_aspect():
                            # print(body.name, shapeNode)
                            if renderType != RENDER_SHADOW:
                                color = numpy.array(shapeNode.visual_aspect_rgba())*255
                                # if color[0] != 0 or color[1] != 0 or color[2] != 0:
                                if sum(self.totalColor) == 765:
                                    c = [int(color[0]), int(color[1]), int(color[2]), int(color[3])]
                                    glColor4ubv(c)
                                else:
                                    glColor3ubv(self.totalColor)

                                self.renderShapeNode(shapeNode)
                            elif body.name != 'ground':
                                glColor3ub(90, 90, 90)
                                self.renderShapeNode(shapeNode)
                    glPopMatrix()
            self.world.render_contacts()

    def renderShapeNode(self, shapeNode):
        """

        :type shapeNode: pydart.ShapeNode
        :return:
        """
        # names = ["BOX", "ELLIPSOID", "CYLINDER", "PLANE",
        #          "MESH", "SOFT_MESH", "LINE_SEGMENT"]

        # shapeNode.shape.render()
        geomType = shapeNode.shape.shape_type_name()
        glPushMatrix()
        glMultMatrixd(shapeNode.relative_transform().transpose())

        # print(geomType)
        if geomType == 'BoxShape':
            shape = shapeNode.shape # type: pydart.BoxShape
            data = shape.size()
            glPushMatrix()
            glTranslatef(-data[0]/2., -data[1]/2., -data[2]/2.)
            self.rc.drawBox(data[0], data[1], data[2])
            glPopMatrix()
        elif geomType == 'CylinderShape':
            shape = shapeNode.shape # type: pydart.CylinderShape
            data = [shape.getRadius(), shape.getHeight()]
            glTranslatef(0., 0., -data[1]/2.)
            self.rc.drawCylinder(data[0], data[1])
            # self.rc.drawCapsule(data[0], data[1])
        elif geomType == 'SphereShape':
            shape = shapeNode.shape  # type: pydart.SphereShape
            data = shape.radius()
            glScalef(data, data, data)
            self.rc.drawSphere(1.)
        elif geomType == 'EllipsoidShape':
            shape = shapeNode.shape  # type: pydart.EllipsoidShape
            data = shape.size()  # type: numpy.ndarray
            glScalef(data[0]/2., data[1]/2., data[2]/2.)
            self.rc.drawSphere(1.)
        glPopMatrix()

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1 or not self.savable:
            self.renderState(self.getState(), renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def getState(self):
        state = []
        for skeleton in self.world.skeletons:
            for body in skeleton.bodynodes:
                body_name = body.name
                bodyFrame = body.world_transform()
                for shapeNode in body.shapenodes:
                    if shapeNode.has_visual_aspect():
                        color = None
                        if sum(self.totalColor) == 765:
                            c = numpy.array(shapeNode.visual_aspect_rgba())*255
                            color = [ int(c[0]), int(c[1]), int(c[2])]
                        else:
                            color = self.totalColor

                        geomT = numpy.dot(bodyFrame, shapeNode.relative_transform())
                        # geomT = bodyFrame
                        geomType = shapeNode.shape.shape_type_name()
                        shape = shapeNode.shape
                        data = None
                        if geomType[0] == 'B':
                            data = shape.size()
                        elif geomType[0] == 'C':
                            data = [shape.radius(), shape.height()]
                        elif geomType[0] == 'E':
                            data = shape.size()
                        elif geomType[0] == 'S':
                            data = shape.radius()
                        state.append((body_name, geomType, geomT, data, color))
        return state

    def renderState(self, state, renderType=RENDER_OBJECT):
        """

        :type state: list[tuple[str, str, numpy.ndarray, numpy.ndarray, tuple]]
        :return:
        """
        glLineWidth(self._lineWidth)

        for elem in state:
            body_name, geomType, geomT, data, color = elem
            glPushMatrix()
            glMultMatrixd(geomT.transpose())

            if renderType != RENDER_SHADOW:
                glColor3ubv(color)
            else:
                #glColor3ubv(shadow_color)
                glColor4ub(shadow_color[0], shadow_color[1], shadow_color[2], 60)

            if geomType[0] == 'B':
                glTranslatef(-data[0]/2., -data[1]/2., -data[2]/2.)
                self.rc.drawBox(data[0], data[1], data[2])
            elif geomType[0] == 'C':
                glTranslatef(0., 0., -data[1]/2.)
                self.rc.drawCylinder(data[0], data[1])
                # self.rc.drawCapsule(data[0], data[1])
            elif geomType[0] == 'E':
                glScalef(data[0]/2., data[1]/2., data[2]/2.)
                self.rc.drawSphere(1.)
            elif geomType[0] == 'S':
                glScalef(data, data, data)
                self.rc.drawSphere(1.)

            glPopMatrix()


class DartModelRenderer(Renderer):
    """
    :type model: cdm.DartModel
    """
    def __init__(self, target, color=(255,255,255), polygonStyle=POLYGON_FILL, lineWidth=1., save_state=True):
        Renderer.__init__(self, target, color)
        self.model = target
        self.rc.setPolygonStyle(polygonStyle)
        self._lineWidth = lineWidth
        self.savable = save_state

    def render(self, renderType=RENDER_OBJECT):
        glLineWidth(self._lineWidth)

        if renderType == RENDER_SHADOW:
            glColor3ub(90, 90, 90)
        else:
            glColor3ubv(self.totalColor)

        for body in self.model.skeleton.bodynodes:
            glPushMatrix()
            glMultMatrixd(body.world_transform().transpose())
            for shapeNode in body.shapenodes:
                if shapeNode.has_visual_aspect():
                    # print(body.name, shapeNode)
                    if renderType != RENDER_SHADOW:
                        color = numpy.array(shapeNode.visual_aspect_rgba())*255
                        # if color[0] != 0 or color[1] != 0 or color[2] != 0:
                        if sum(self.totalColor) == 765:
                            c = [ int(color[0]), int(color[1]), int(color[2]), int(color[3]) ]
                            glColor4ubv(c)
                        else:
                            glColor3ubv(self.totalColor)
                    else:
                        glColor3ub(90, 90, 90)


                    self.renderShapeNode(shapeNode)
            glPopMatrix()
        self.model.world.render_contacts()

    def renderShapeNode(self, shapeNode):
        """

        :type shapeNode: pydart.ShapeNode
        :return:
        """
        # names = ["BOX", "ELLIPSOID", "CYLINDER", "PLANE",
        #          "MESH", "SOFT_MESH", "LINE_SEGMENT"]

        # shapeNode.shape.render()
        geomType = shapeNode.shape.shape_type_name()
        glPushMatrix()
        glMultMatrixd(shapeNode.relative_transform().transpose())
        if geomType == 'BoxShape':
            shape = shapeNode.shape # type: pydart.BoxShape
            data = shape.size()
            glPushMatrix()
            glTranslatef(-data[0]/2., -data[1]/2., -data[2]/2.)
            self.rc.drawBox(data[0], data[1], data[2])
            glPopMatrix()
        elif geomType == 'CylinderShape':
            shape = shapeNode.shape # type: pydart.CylinderShape
            data = [shape.getRadius(), shape.getHeight()]
            glTranslatef(0., 0., -data[1]/2.)
            self.rc.drawCylinder(data[0], data[1])
            # self.rc.drawCapsule(data[0], data[1])
        elif geomType == 'EllipsoidShape':
            shape = shapeNode.shape  # type: pydart.EllipsoidShape
            data = shape.size()  # type: numpy.ndarray
            glScalef(data[0]/2., data[1]/2., data[2]/2.)
            self.rc.drawSphere(1.)
        glPopMatrix()

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1 or not self.savable:
            self.renderState(self.getState(), renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def getState(self):
        state = []
        for body in self.model.skeleton.bodynodes:
            bodyFrame = body.world_transform()
            body_name = body.name
            for shapeNode in body.shapenodes:
                if shapeNode.has_visual_aspect():
                    color = None
                    if sum(self.totalColor) == 765:
                        c = numpy.array(shapeNode.visual_aspect_rgba())*255
                        color = [ int(c[0]), int(c[1]), int(c[2])]
                    else:
                        color = self.totalColor

                    # geomT = numpy.dot(bodyFrame, shapeNode.relative_transform())
                    geomT = bodyFrame
                    geomType = shapeNode.shape.shape_type_name()
                    shape = shapeNode.shape
                    data = None
                    if geomType[0] == 'B':
                        data = shape.size()
                    elif geomType[0] == 'C':
                        data = [shape.radius(), shape.height()]
                    elif geomType[0] == 'E':
                        data = shape.size()
                    state.append((geomType, geomT, data, color))
        return state

    def renderState(self, state, renderType=RENDER_OBJECT):
        """

        :type state: list[tuple[str, numpy.ndarray, numpy.ndarray, tuple]]
        :return:
        """
        glLineWidth(self._lineWidth)

        for elem in state:
            geomType, geomT, data, color = elem
            glPushMatrix()
            glMultMatrixd(geomT.transpose())

            if renderType != RENDER_SHADOW:
                glColor3ubv(color)
            else:
                glColor3ub(90, 90, 90)

            if geomType[0] == 'B':
                glTranslatef(-data[0]/2., -data[1]/2., -data[2]/2.)
                self.rc.drawBox(data[0], data[1], data[2])
            elif geomType[0] == 'C':
                glTranslatef(0., 0., -data[1]/2.)
                self.rc.drawCylinder(data[0], data[1])
                # self.rc.drawCapsule(data[0], data[1])
            elif geomType[0] == 'E':
                glScalef(data[0]/2., data[1]/2., data[2]/2.)
                self.rc.drawSphere(1.)

            glPopMatrix()


class JointMotionRenderer(Renderer):
    def __init__(self, target, color=(0,255,255), linkStyle=LINK_LINE, lineWidth=1.):
        Renderer.__init__(self, target, color)
        self.motion = target
        self.renderFrames = None
        self.setLinkStyle(linkStyle)
        self.rc.setLineWidth(lineWidth)

    def setLinkStyle(self, linkStyle):
        self.linkStyle = linkStyle
        if self.linkStyle == LINK_WIREBOX:
            self.rc.setPolygonStyle(POLYGON_LINE)
        else:
            self.rc.setPolygonStyle(POLYGON_FILL)

    def render(self, renderType=RENDER_OBJECT):
        if len(self.motion) > 0:
            self.rc.beginDraw()
            if renderType == RENDER_SHADOW:
                glColor3ubv(self.shadowColor)
            else:
                glColor3ubv(self.totalColor)
            if self.renderFrames is None:
                posture = self.motion[self.motion.frame]
                self.renderJointPosture(posture)
            else:
                for renderFrame in self.renderFrames:
                    posture = self.motion[renderFrame]
                    self.renderJointPosture(posture)

    def renderJointPosture(self, posture):
        joint = posture.skeleton.root
        glPushMatrix()
        # glTranslatef(posture.rootPos[0], posture.rootPos[1], posture.rootPos[2])
        self._renderJoint(joint, posture)
        glPopMatrix()

    def _renderJoint(self, joint, posture):
        glPushMatrix()
        glTranslatef(joint.offset[0], joint.offset[1], joint.offset[2])
#        glMultMatrixf(mm.R2T(posture.localRMap[joint.name]).transpose())
        glMultMatrixf(mm.p2T(posture.local_ts[posture.skeleton.getElementIndex(joint.name)]).transpose())
        glMultMatrixf(mm.R2T(posture.localRs[posture.skeleton.getElementIndex(joint.name)]).transpose())

#        if joint.name in self.partColors:
#            color = self.partColors[joint.name]
#        else:
#            color = self.totalColor
            
        if joint == self.selectedElement:
            glColor3ubv(SELECTION_COLOR)
            ygh.beginDraw()
            ygh.drawCoordinate()
            ygh.endDraw()
            
        # 1
#        ygh.drawPoint((0,0,0), color)

        if self.linkStyle == LINK_LINE:
            self.rc.drawPoint((0,0,0))
            for childJoint in joint.children:
                self.rc.drawLine((0,0,0), childJoint.offset)
                
        elif self.linkStyle == LINK_BONE:
#            self.rc.drawPoint((0,0,0))
            self.rc.drawLine((-.05,0,0), (.05,0,0))
            for childJoint in joint.children:
                self.rc.drawLine((0,0,0), childJoint.offset)
        
        elif self.linkStyle == LINK_SOLIDBOX or self.linkStyle == LINK_WIREBOX:        
            if len(joint.children) > 0:
                glPushMatrix()
                
                offset = numpy.array([0.,0.,0.])
                for childJoint in joint.children:
                    offset += childJoint.offset
                offset = offset/len(joint.children)
                
                defaultBoneV = numpy.array([0,0,mm.length(offset)])
                boneT = mm.R2T(mm.getSO3FromVectors(defaultBoneV, offset))
                glMultMatrixf(boneT.transpose())
        
                glTranslatef(-.05,-.05,0)
                # ygh.beginDraw()
                # ygh.drawCoordinate()
                # ygh.endDraw()
        
                self.rc.drawBox(.1,.1,mm.length(offset))
                glPopMatrix()
        
        if joint == self.selectedElement:
            glColor3ubv(self.totalColor)
        
        for childJoint in joint.children:
            self._renderJoint(childJoint, posture)
        glPopMatrix()

    # def renderState(self, state, renderType=RENDER_OBJECT):
    #     if len(self.motion) > 0:
    #         zeroVec = (0., 0., 0.)
    #         glPushMatrix()
    #         glTranslatef(state[0][0], state[0][1], state[0][2])
    #         for stateIdx in range(1, len(state)):
    #             # print(len(state[stateIdx]))
    #             # print(state[stateIdx][0], state[stateIdx][2])
    #             # print(state[stateIdx][3])
    #             jointname, jointoffset, jointT, childrenOffsets = state[stateIdx]
    #             glTranslatef(jointoffset[0], jointoffset[1], jointoffset[2])
    #             glMultMatrixf(jointT.transpose())
    #
    #             if self.selectedElement is not None and jointname == self.selectedElement.name:
    #                 glColor3ubv(SELECTION_COLOR)
    #                 ygh.beginDraw()
    #                 ygh.drawCoordinate()
    #                 ygh.endDraw()
    #
    #             if self.linkStyle == LINK_LINE:
    #                 self.rc.drawPoint(zeroVec)
    #                 for childrenOffset in childrenOffsets:
    #                     self.rc.drawLine(zeroVec, childrenOffset)
    #
    #             elif self.linkStyle == LINK_BONE:
    #                 self.rc.drawLine((-.05, 0., 0.), (.05, 0., 0.))
    #                 for childrenOffset in childrenOffsets:
    #                     self.rc.drawLine(zeroVec, childrenOffset)
    #
    #             elif self.linkStyle in (LINK_SOLIDBOX, LINK_WIREBOX):
    #                 if len(childrenOffsets) > 0:
    #                     offset = sum(childrenOffsets)/len(childrenOffsets)
    #                     defaultBoneV = numpy.array([0., 0., mm.length(offset)])
    #                     boneT = mm.R2T(mm.getSO3FromVectors(defaultBoneV, offset))
    #                     glPushMatrix()
    #                     glMultMatrixf(boneT.transpose())
    #                     glTranslatef(-.05, -.05, 0.)
    #                     self.rc.drawBox(.1, .1, mm.length(offset))
    #                     glPopMatrix()
    #
    #             if self.selectedElement is not None:
    #                 if self.selectedElement.name is not None and jointname == self.selectedElement.name:
    #                     glColor3ubv(self.totalColor)
    #         glPopMatrix()
    #
    # def renderFrame(self, frame, renderType=RENDER_OBJECT):
    #     if len(self.motion) > 0:
    #         if self.renderFrames is None:
    #             self.renderState(self.savedState[frame], renderType)
    #         else:
    #             for renderFrame in self.renderFrames:
    #                 posture = self.motion[renderFrame]
    #                 self.renderJointPosture(posture)
    #
    # def getState(self):
    #     def _getState(_posture, joint, parentJointT):
    #         offset = copy.deepcopy(joint.offset)
    #         # jointT = numpy.dot(parentJointT, mm.R2T(_posture.localRs[_posture.skeleton.getElementIndex(joint.name)]))
    #         jointT = numpy.dot(parentJointT, mm.R2T(_posture.localRs[_posture.skeleton.getElementIndex(joint.name)]))
    #         childrenOffsets = []
    #         for child in joint.children:
    #             childrenOffsets.append(copy.deepcopy(child.offset))
    #
    #         _state = [[joint.name, offset, jointT, childrenOffsets]]
    #
    #         for child in joint.children:
    #             # _state.append(_getState(_posture, child, jointT))
    #             _state.extend(_getState(_posture, child, jointT))
    #
    #         return _state
    #
    #     if self.motion.frame >= 0 and len(self.motion) > 0:
    #         posture = self.motion[self.motion.frame]
    #         state = [[posture.rootPos[0], posture.rootPos[1], posture.rootPos[2]]]
    #         state.extend(_getState(posture, posture.skeleton.root, numpy.eye(4)))
    #
    #         return state


class BasicSkeletonRenderer(Renderer):
    def __init__(self, Ts, color=(255, 255, 255), offset_draw=(0., 0., 0.)):
        """
        Ts should be dict type.
        pelvis, spine_ribs, head, thigh_R, shin_R, foot_R, upper_limb_R, lower_limb_R
                                  thigh_L, shin_L, foot_L, upper_limb_L, lower_limb_L
        :param Ts:
        :param color:
        """
        REAL_JOINT = True

        from glob import glob
        Renderer.__init__(self, Ts, color)
        self.Ts_init = Ts
        # self.savedState.append(Ts)
        self.objs = dict()  # type: dict[str, ObjImporter]

        for path in glob('../../data/obj/zygote_skeleton_common/*.obj'):
            filename = path.split('/')[-1].split('.')[0]
            self.objs[filename] = ObjImporter()
            self.objs[filename].import_obj(path, 0.01)

        if REAL_JOINT:
            for path in glob('../../data/obj/zygote_skeleton_real_joint/*.obj'):
                filename = path.split('/')[-1].split('.')[0]
                self.objs[filename] = ObjImporter()
                self.objs[filename].import_obj(path, 0.01)
        else:
            for path in glob('../../data/obj/zygote_skeleton_basic/*.obj'):
                filename = path.split('/')[-1].split('.')[0]
                self.objs[filename] = ObjImporter()
                self.objs[filename].import_obj(path, 0.01)

        self.offset = dict()  # type: dict[str, numpy.ndarray]
        self.offset['pelvis'] = numpy.array([0., 0., 0.])
        self.offset['spine_ribs'] = numpy.array([0., 0.0577, -0.01791])
        self.offset['head'] = numpy.array([0., 0.57875, 0.04319])
        self.offset['upper_limb_R'] = numpy.array([-0.19431, 0.40374, 0.01608])
        self.offset['lower_limb_R'] = numpy.array([-0.327, 0.01339, -0.0251])
        self.offset['upper_limb_L'] = numpy.array([0.19431, 0.40374, 0.01608])
        self.offset['lower_limb_L'] = numpy.array([0.327, 0.01339, -0.0251])

        self.offset['thigh_R'] = numpy.array([-0.08931, -0.031, 0.01779])
        self.offset['shin_R'] = numpy.array([-0.007, -0.40402, -0.00173])
        self.offset['foot_R'] = numpy.array([0.01482, -0.46019, -0.02403])
        self.offset['foot_heel_R'] = numpy.array([0.01482, -0.46019, -0.02403])

        self.offset['thigh_L'] = numpy.array([0.08931, -0.031, 0.01779])
        self.offset['shin_L'] = numpy.array([0.007, -0.40402, -0.00173])
        self.offset['foot_L'] = numpy.array([-0.01482, -0.46019, -0.02403])
        self.offset['foot_heel_L'] = numpy.array([-0.01482, -0.46019, -0.02403])

        if REAL_JOINT:
            self.offset['heel_R'] = numpy.array([0., 0., 0.])
            self.offset['outside_metatarsal_R'] = numpy.array([-0.02784, -0.03463, 0.0452])
            self.offset['outside_phalanges_R'] = numpy.array([-0.00773, -0.01936, 0.05877])
            self.offset['inside_phalanges_R'] = numpy.array([-0.01823, -0.05399, 0.10397])

            self.offset['heel_L'] = numpy.array([0., 0., 0.])
            self.offset['outside_metatarsal_L'] = numpy.array([0.02784, -0.03463, 0.0452])
            self.offset['outside_phalanges_L'] = numpy.array([0.00773, -0.01936, 0.05877])
            self.offset['inside_phalanges_L'] = numpy.array([0.01823, -0.05399, 0.10397])
        else:
            self.offset['outside_metatarsal_R'] = numpy.array([-0.02714, -0.05689, 0.])
            self.offset['outside_phalanges_R'] = numpy.array([0., 0., 0.09059])
            self.offset['inside_metatarsal_R'] = numpy.array([0.01409, 0., 0.])
            self.offset['inside_phalanges_R'] = numpy.array([0., 0.00762, 0.11214])

            self.offset['outside_metatarsal_L'] = numpy.array([0.02714, -0.05689, 0.])
            self.offset['outside_phalanges_L'] = numpy.array([0., 0., 0.09059])
            self.offset['inside_metatarsal_L'] = numpy.array([-0.01409, 0., 0.])
            self.offset['inside_phalanges_L'] = numpy.array([0., 0.00762, 0.11214])

        self.children = dict()
        self.children['pelvis'] = ['spine_ribs', 'thigh_R', 'thigh_L']
        self.children['spine_ribs'] = ['head', 'upper_limb_R', 'upper_limb_L']
        self.children['upper_limb_R'] = ['lower_limb_R']
        self.children['lower_limb_R'] = []
        self.children['upper_limb_L'] = ['lower_limb_L']
        self.children['lower_limb_L'] = []
        self.children['head'] = []

        self.children['thigh_R'] = ['shin_R']
        self.children['shin_R'] = ['foot_R']
        self.children['foot_R'] = []

        self.children['thigh_L'] = ['shin_L']
        self.children['shin_L'] = ['foot_L']
        self.children['foot_L'] = []

        if REAL_JOINT:
            self.children['foot_R'] = ['heel_R', 'outside_metatarsal_R', 'inside_phalanges_R']
            self.children['heel_R'] = []
            self.children['outside_metatarsal_R'] = ['outside_phalanges_R']
            self.children['outside_phalanges_R'] = []
            self.children['inside_phalanges_R'] = []

            self.children['foot_L'] = ['heel_L', 'outside_metatarsal_L', 'inside_phalanges_L']
            self.children['heel_L'] = []
            self.children['outside_metatarsal_L'] = ['outside_phalanges_L']
            self.children['outside_phalanges_L'] = []
            self.children['inside_phalanges_L'] = []
        else:
            self.children['shin_R'] = ['foot_heel_R']
            self.children['foot_heel_R'] = ['outside_metatarsal_R']
            self.children['outside_metatarsal_R'] = ['inside_metatarsal_R', 'outside_phalanges_R']
            self.children['outside_phalanges_R'] = []
            self.children['inside_metatarsal_R'] = ['inside_phalanges_R']
            self.children['inside_phalanges_R'] = []

            self.children['shin_L'] = ['foot_heel_L']
            self.children['foot_heel_L'] = ['outside_metatarsal_L']
            self.children['outside_metatarsal_L'] = ['inside_metatarsal_L', 'outside_phalanges_L']
            self.children['outside_phalanges_L'] = []
            self.children['inside_metatarsal_L'] = ['inside_phalanges_L']
            self.children['inside_phalanges_L'] = []

        def get_offset_from_root(offset_dict, child_dict, link_name):
            """

            :param offset_dict:
            :param child_dict:
            :type child_dict: dict
            :param link_name:
            :return:
            """
            parent_dict = dict()
            for parent, child_list in child_dict.items():
                for child in child_list:
                    parent_dict[child] = parent

            offset = numpy.zeros(3)
            while link_name != 'pelvis':
                offset = offset + offset_dict[link_name]
                link_name = parent_dict[link_name]

            return offset

        # for link_name in self.objs.keys():
        #     print(link_name, get_offset_from_root(self.offset, self.children, link_name))

        self.offset_draw = copy.deepcopy(offset_draw)

        self.isInit = False

    def appendFrameState(self, Ts, body_color={}):
        self.savedState.append((Ts, body_color))

    def render(self, renderType=RENDER_OBJECT):
        print("Renderer.render() : Must subclass me")
        raise NotImplementedError

    def renderState(self, state, renderType=RENDER_OBJECT):
        """

        :param state:
        :type state: tuple[dict[str, np.ndarray], dict[str, tuple[int, int, int]]]
        :param renderType:
        :return:
        """
        # glEnable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(self.offset_draw[0], self.offset_draw[1], self.offset_draw[2])
        self._render('pelvis', state, renderType)
        glPopMatrix()
        # glDisable(GL_LIGHTING)

    def _render(self, body_name, state, renderType=RENDER_OBJECT):
        glPushMatrix()
        offset = self.offset[body_name]
        glTranslatef(offset[0], offset[1], offset[2])
        glMultMatrixf(state[0][body_name].T)
        if renderType == RENDER_OBJECT:
            if body_name in state[1].keys():
                glColor3ubv(state[1][body_name])
            else:
                glColor3ubv(self.totalColor)
        elif renderType == RENDER_SHADOW:
            # glColor3ub(90, 90, 90)
            glColor3ubv(shadow_color)
        if FOOT_RENDER_ONLY:
            if LEFT_FOOT_ONLY:
                if body_name in ('foot_heel_L', 'foot_L', 'heel_L', 'outside_metatarsal_L', 'outside_phalanges_L', 'inside_phalanges_L'):
                    self.objs[body_name].draw()
            elif RIGHT_FOOT_ONLY:
                if body_name in ('foot_heel_R', 'foot_R', 'heel_R', 'outside_metatarsal_R', 'outside_phalanges_R', 'inside_phalanges_R'):
                    self.objs[body_name].draw()
            else:
                if body_name in ('foot_heel_R', 'foot_R', 'heel_R', 'outside_metatarsal_R', 'outside_phalanges_R', 'inside_phalanges_R',
                                 'foot_heel_L', 'foot_L', 'heel_L', 'outside_metatarsal_L', 'outside_phalanges_L', 'inside_phalanges_L'):
                    self.objs[body_name].draw()
        else:
            self.objs[body_name].draw()
        for child_name in self.children[body_name]:
            self._render(child_name, state, renderType)
        glPopMatrix()

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1:
            self.renderState((self.Ts_init, dict()), renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    # def saveState(self):
    #     self.savedState.append(self.getState())

    def get_max_saved_frame(self):
        return len(self.savedState) - 1

    def set_offset_Y(self, offset_Y):
        self.offset_Y = offset_Y


class PointMotionRenderer(Renderer):
    def __init__(self, target, color = (0,0,255)):
        Renderer.__init__(self, target, color)
        self.motion = target
    def render(self, renderType=RENDER_OBJECT):
        glColor3ubv(self.totalColor)
        posture = self.motion[self.motion.frame]
        self.renderPointPosture(posture)
    def renderPointPosture(self, posture):
        for point in posture.positions:
            #            if name in self.partColors:
            #                glColor3ubv(self.partColors[name])
            #            else:
            #                glColor3ubv(self.totalColor)
            glPushMatrix()

            if point == self.selectedElement:
                glColor3ubv(SELECTION_COLOR)

            self.rc.drawPoint(point)

            if point == self.selectedElement:
                glColor3ubv(self.totalColor)

            glPopMatrix()

class MMMotionRenderer(Renderer):
    def __init__(self, target, color = (0,0,255)):
        Renderer.__init__(self, target, color)
        self.motion = target
    def render(self, renderType=RENDER_OBJECT):
        glColor3ubv(self.totalColor)
        posture = self.motion[self.motion.frame]
        self.renderPointPosture(posture)
    def renderPointPosture(self, posture):
        for name, point in posture.pointMap.items():
            #            if name in self.partColors:
            #                glColor3ubv(self.partColors[name])
            #            else:
            #                glColor3ubv(self.totalColor)
            glPushMatrix()

            if point == self.selectedElement:
                glColor3ubv(SELECTION_COLOR)

            self.rc.drawPoint(point)

            if point == self.selectedElement:
                glColor3ubv(self.totalColor)

            glPopMatrix()
        for link in posture.skeleton.links:
            self.rc.drawLine(posture.pointMap[link[0]], posture.pointMap[link[1]])


class MeshRenderer(Renderer):
    def __init__(self, mesh, color = (127,127,127), drawStyle = POLYGON_LINE):
        Renderer.__init__(self, mesh, color)
        self.mesh = mesh
        self.rc.setPolygonStyle(drawStyle)
    def render(self, renderType=RENDER_OBJECT):
        if isinstance(self.selectedElement, yms.Vertex):
            glColor3ubv(SELECTION_COLOR)
            self.rc.drawPoint(self.selectedElement.pos)

        pmid = None
        glPolygonMode(GL_FRONT, GL_LINE)
        glColor3ubv(self.totalColor)
        glBegin(GL_TRIANGLES)
        for f in self.mesh.faces:
            if f == self.selectedElement:
                glColor3ubv(SELECTION_COLOR)

            p0 = self.mesh.vertices[f.vertexIndex[0]].pos
            p1 = self.mesh.vertices[f.vertexIndex[1]].pos
            p2 = self.mesh.vertices[f.vertexIndex[2]].pos
            glVertex3f(p0[0], p0[1], p0[2])
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])

            if f == self.selectedElement:
                pmid = (p0+p1+p2)/3.
                glColor3ubv(self.totalColor)
        glEnd()

        if pmid!=None:
            glColor3ubv(SELECTION_COLOR)
            self.rc.drawPoint(pmid)

#===============================================================================
# # debugging renderers
#===============================================================================
class PointsRenderer(Renderer):
    def __init__(self, points, color=(255,0,0), pointStyle=POINT_CROSS, save_state=True):
        Renderer.__init__(self, points, color)
        self.points = points
        self.pointStyle = pointStyle
        self.rc.setLineWidth(2.)
        self.savable = save_state

    def render(self, renderType=RENDER_OBJECT):
        if renderType == RENDER_OBJECT:
            self.rc.beginDraw()
            glColor3ubv(self.totalColor)
            for point in self.points:
                if point is not None:
                    if self.pointStyle==POINT_POINT:
                        self.rc.drawPoint(point)
                    elif self.pointStyle==POINT_CROSS:
                        self.rc.drawCross(point)
                    elif self.pointStyle==POINT_CUBE:
                        self.rc.drawCube(point)

    def renderState(self, state, renderType=RENDER_OBJECT):
        if renderType == RENDER_OBJECT:
            self.rc.beginDraw()
            glColor3ubv(self.totalColor)
            for point in state:
                if point is not None:
                    if self.pointStyle==POINT_POINT:
                        self.rc.drawPoint(point)
                    elif self.pointStyle==POINT_CROSS:
                        self.rc.drawCross(point)
                    elif self.pointStyle==POINT_CUBE:
                        self.rc.drawCube(point)

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1 or not self.savable:
            self.renderState(self.getState(), renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def getState(self):
        return copy.deepcopy(self.points)


class LinesRenderer(Renderer):
    def __init__(self, points, color = (255,0,0), lineWidth=.02, save_state=True):
        Renderer.__init__(self, points, color)
        self.points = points
        self.lineWidth = lineWidth
        self.rc.setLineWidth(lineWidth)
        self.savable = save_state

    def render(self, renderType=RENDER_OBJECT):
        if renderType==RENDER_OBJECT:
            glColor3ubv(self.totalColor)
            self.rc.beginDraw()
            for i in range(len(self.points)-1):
                if self.points[i] is not None:
                    point1 = self.points[i]
                    point2 = self.points[i+1]
                    self.rc.drawLine(point1, point2)

    def renderState(self, state, renderType=RENDER_OBJECT):
        if renderType == RENDER_OBJECT:
            glColor3ubv(self.totalColor)
            self.rc.beginDraw()
            for i in range(len(state)-1):
                if state[i] is not None:
                    point1 = state[i]
                    point2 = state[i+1]
                    self.rc.drawLine(point1, point2)

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1 or not self.savable:
            self.renderState(self.getState(), renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def getState(self):
        return copy.deepcopy(self.points)

class VectorsRenderer(Renderer):
    def __init__(self, vectors, origins, color = (255,0,0), lineWidth=.02, name=""):
        Renderer.__init__(self, vectors, color)
        self.vectors = vectors
        self.origins = origins
        self.lineWidth = lineWidth
        self.rc.setLineWidth(lineWidth)
        self.name = name

    def render(self, renderType=RENDER_OBJECT):
        if renderType==RENDER_OBJECT:
            glColor3ubv(self.totalColor)
            self.rc.beginDraw()
            for i in range(len(self.vectors)):
                if (self.vectors[i] is not None) and (self.origins[i] is not None):
                    origin = self.origins[i]; vector = self.vectors[i]
                    self.rc.drawLine(origin, (origin[0]+vector[0],origin[1]+vector[1],origin[2]+vector[2]))


    def renderState(self, state, renderType=RENDER_OBJECT):
        if renderType == RENDER_OBJECT:
            self.rc.beginDraw()
            glColor3ubv(self.totalColor)
            vectors = state[0]
            origins = state[1]

            for i in range(len(vectors)):
                if (vectors[i] is not None) and (origins[i] is not None):
                    origin, vector = origins[i], vectors[i]
                    self.rc.drawLine(origin, (origin[0]+vector[0],origin[1]+vector[1],origin[2]+vector[2]))

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1:
            self.renderState(self.getState(), renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def getState(self):
        return [copy.deepcopy(self.vectors), copy.deepcopy(self.origins)]



class PolygonRenderer(Renderer):
    def __init__(self, vertices, color = (0,255,0)):
        Renderer.__init__(self, vertices, color)
        self.points = vertices
        if len(self.points) == 3:
            self.polygonMode = GL_TRIANGLES
        else:
            self.polygonMode = GL_QUADS

    def render(self, renderType=RENDER_OBJECT):
        glColor3ubv(self.totalColor)
        glDisable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_LINE)
        glBegin(self.polygonMode)
        for v in self.points:
            glVertex3fv(v)
        glEnd()
        glEnable(GL_CULL_FACE)

class FramesRenderer(Renderer):
    def __init__(self, Ts, color = (0,255,0), axisLength = .5):
        Renderer.__init__(self, Ts, color)
        self.Ts = Ts
        self.axisLength = axisLength

    def render(self, renderType=RENDER_OBJECT):
        if renderType == RENDER_SHADOW:
            return

        for T in self.Ts:
            if T is not None:
                glPushMatrix()
                glMultMatrixf(T.transpose())
                ygh.drawCoordinate(self.totalColor, self.axisLength)
                glPopMatrix()
                #        R, p = mm.T2Rp(self.T)
                #        axes = R.transpose()
                #        ygh.drawVector(axes[0], p, (255,0,0))
                #        ygh.drawVector(axes[1], p, (0,255,0))
                #        ygh.drawVector(axes[2], p, (0,0,255))

class OrientationsRenderer(Renderer):
    def __init__(self, Rs, ps, color = (0,255,0), axisLength = .5):
        Renderer.__init__(self, Rs, color)
        self.Rs = Rs
        self.ps = ps
        self.axisLength = axisLength
    def render(self, renderType=None):
        if renderType != RENDER_SHADOW:
            for i in range(len(self.Rs)):
                if self.Rs[i] is not None and self.ps[i] is not None:
                    T = mm.Rp2T(self.Rs[i], self.ps[i])
                    glPushMatrix()
                    glMultMatrixf(T.transpose())
                    ygh.drawCoordinate(self.totalColor, self.axisLength)
                    glPopMatrix()


class ForcesRenderer(Renderer):
    def __init__(self, forces, points, color=(255,0,0), ratio=1., lineWidth=.02, fromPoint=True):
        Renderer.__init__(self, None, color)
        self.forces = forces
        self.points = points
        self.ratio = ratio
        self.lineWidth = lineWidth
        self.fromPoint = fromPoint
        self.rc.setNormalStyle(NORMAL_SMOOTH)

    def render(self, renderType=RENDER_OBJECT):
        if renderType == RENDER_OBJECT:
            self.rc.beginDraw()
            glColor3ubv(self.totalColor)
            for i in range(len(self.forces)):
                if (self.forces[i] is not None) and (self.points[i] is not None):
                    if not self.fromPoint:
                        self.rc.drawArrow(None, self.points[i], mm.v3_scale(self.forces[i], self.ratio), self.lineWidth)
                    else:
                        self.rc.drawArrow(self.points[i], None, mm.v3_scale(self.forces[i], self.ratio), self.lineWidth)

class WideArrowRenderer(Renderer):
    def __init__(self, forces, points, color=(255,0,0), ratio=1., lineWidth=.02, heightRatio=.2, fromPoint=True, polygonStyle=POLYGON_FILL):
        Renderer.__init__(self, None, color)
        self.forces = forces
        self.points = points
        self.ratio = ratio
        self.lineWidth = lineWidth
        self.fromPoint = fromPoint
        self.heightRatio = heightRatio
        self.rc.setPolygonStyle(polygonStyle)
        self.rc.setNormalStyle(NORMAL_SMOOTH)
        self.state_init = [[None], [None]]

    def render(self, renderType=RENDER_OBJECT):
        if renderType == RENDER_OBJECT:
            self.rc.beginDraw()
            glColor3ubv(self.totalColor)
            for i in range(len(self.forces)):
                if self.forces[i] is not None and self.points[i] is not None:
                    glPushMatrix()
                    # glScalef(1, self.heightRatio, 1)
                    if self.fromPoint==False:
                        self.rc.drawArrow(None, self.points[i], mm.v3_scale(self.forces[i], self.ratio), self.lineWidth)
                    else:
                        self.rc.drawArrow(self.points[i], None, mm.v3_scale(self.forces[i], self.ratio), self.lineWidth)
                    glPopMatrix()

    def getState(self):
        return [copy.deepcopy(self.forces), copy.deepcopy(self.points)]

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1:
            self.renderState(self.state_init, renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def renderState(self, state, renderType=RENDER_OBJECT):
        forces, points = state[0], state[1]
        if renderType == RENDER_OBJECT:
            self.rc.beginDraw()
            glColor3ubv(self.totalColor)
            for i in range(len(forces)):
                if forces[i] is not None and points[i] is not None:
                    glPushMatrix()
                    # glScalef(1, self.heightRatio, 1)
                    if self.fromPoint==False:
                        self.rc.drawArrow(None, points[i], mm.v3_scale(forces[i], self.ratio), self.lineWidth)
                    else:
                        self.rc.drawArrow(points[i], None, mm.v3_scale(forces[i], self.ratio), self.lineWidth)
                    glPopMatrix()

class JointArrowRenderer(Renderer):
    def __init__(self, forces, points, color=(255,0,0), ratio=1., lineWidth=.02, heightRatio=.2, fromPoint=True, polygonStyle=POLYGON_FILL):
        Renderer.__init__(self, None, color)
        self.forces = forces
        self.points = points
        self.ratio = ratio
        self.lineWidth = lineWidth
        self.fromPoint = fromPoint
        self.heightRatio = heightRatio
        self.rc.setPolygonStyle(polygonStyle)
        self.rc.setNormalStyle(NORMAL_SMOOTH)
        self.state_init = [[None], [None]]

    def render(self, renderType=RENDER_OBJECT):
        if renderType == RENDER_OBJECT:
            self.rc.beginDraw()
            glColor3ubv(self.totalColor)
            for i in range(len(self.forces)):
                if self.forces[i] is not None and self.points[i] is not None:
                    glPushMatrix()
                    # glScalef(1, self.heightRatio, 1)
                    if self.fromPoint==False:
                        self.rc.drawArrow(None, self.points[i], mm.v3_scale(self.forces[i], self.ratio), self.lineWidth)
                    else:
                        self.rc.drawArrow(self.points[i], None, mm.v3_scale(self.forces[i], self.ratio), self.lineWidth)
                    glPopMatrix()

    def getState(self):
        return [copy.deepcopy(self.forces), copy.deepcopy(self.points)]

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1:
            self.renderState(self.state_init, renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def renderState(self, state, renderType=RENDER_OBJECT):
        forces, points = state[0], state[1]
        if renderType == RENDER_OBJECT:
            self.rc.beginDraw()
            glColor3ubv(self.totalColor)
            for i in range(len(forces)):
                if forces[i] is not None and points[i] is not None:
                    glPushMatrix()
                    # glScalef(1, self.heightRatio, 1)
                    if not self.fromPoint:
                        self.rc.drawJointArrow(None, points[i], mm.v3_scale(forces[i], self.ratio), self.lineWidth)
                    else:
                        self.rc.drawJointArrow(points[i], None, mm.v3_scale(forces[i], self.ratio), self.lineWidth)
                    glPopMatrix()


class TorquesRenderer(Renderer):
    def __init__(self, torques, points, color=(255,0,0), ratio=1., lineWidth=.02, radius=.1, fromPoint=True):
        Renderer.__init__(self, None, color)
        self.torques = torques
        self.points = points
        self.ratio = ratio
        self.radius = radius
        self.lineWidth = lineWidth
        self.fromPoint = fromPoint
        self.rc.setNormalStyle(NORMAL_SMOOTH)
    def render(self, renderType=RENDER_OBJECT):
        self.rc.beginDraw()
        glColor3ubv(self.totalColor)
        for i in range(len(self.torques)):
            if self.torques[i]!=None and self.points[i]!=None:
                if self.fromPoint==False:
                    self.rc.drawCircularArrow(None, self.points[i], mm.v3_scale(self.torques[i], self.ratio), self.lineWidth, self.radius)
                else:
                    self.rc.drawCircularArrow(self.points[i], None, mm.v3_scale(self.torques[i], self.ratio), self.lineWidth, self.radius)


class TextRenderer(Renderer):
    def __init__(self, texts: list, pos: list, color=fltk.FL_BLACK, text_size=1):
        Renderer.__init__(self, None, color)
        self.texts = texts
        self.text_pos = pos
        self.text_size = text_size

    def render(self, renderType=RENDER_OBJECT):
        self.renderState(self.getState(), renderType)

    def getState(self):
        return [copy.deepcopy(self.texts), copy.deepcopy(self.text_pos)]

    def renderFrame(self, frame, renderType=RENDER_OBJECT):
        if frame == -1:
            self.renderState(self.getState(), renderType)
        elif frame == self.get_max_saved_frame() + 1:
            self.saveState()
            self.renderState(self.savedState[frame], renderType)
        elif frame <= self.get_max_saved_frame():
            self.renderState(self.savedState[frame], renderType)
        else:
            self.renderState(self.savedState[-1], renderType)

    def renderState(self, state, renderType=RENDER_OBJECT):
        texts, text_pos = state[0], state[1]
        # fltk.gl_start()
        mat_model_view = glGetDoublev(GL_MODELVIEW_MATRIX)
        mat_projection = glGetDoublev(GL_PROJECTION_MATRIX)
        for text_idx, text in enumerate(texts):
            if text is not None:
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                glOrtho(-1., 1., -1., 1., -1., 1.)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                fltk.gl_color(fltk.FL_BLACK)
                fltk.gl_font(0, self.text_size)
                fltk.gl_draw(text, text_pos[text_idx][0], text_pos[text_idx][1])

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(mat_projection)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(mat_model_view)
        # fltk.gl_finish()

class VpBodyRenderer(Renderer):
    # boxSizes[i] = (lx, ly, lz)
    # Ts[i] = SE3
    def __init__(self, body, color=(255,0,0), polygonStyle=POLYGON_FILL, lineWidth=1.):
        Renderer.__init__(self, None, color)
        self.body = body
        self.rc.setPolygonStyle(polygonStyle)
        self.lineWidth = lineWidth
    def render(self, renderType=RENDER_OBJECT):
        self.rc.beginDraw()
        if renderType == RENDER_OBJECT:
            glColor3ubv(self.totalColor)
        else:
            glColor3ubv(self.shadowColor)

        boxSize = self.body.getShape(); T = self.body.getFrame()
        glPushMatrix()
        glMultMatrixf(T.T)
        glLineWidth(self.lineWidth)
        self.rc.drawBox(boxSize[0], boxSize[1], boxSize[2])
        glLineWidth(1.)
        glPopMatrix()

class VpBodiesRenderer(Renderer):
    # boxSizes[i] = (lx, ly, lz)
    # Ts[i] = SE3
    def __init__(self, bodies, color=(255,0,0), polygonStyle=POLYGON_FILL, lineWidth=1.):
        Renderer.__init__(self, None, color)
        self.bodies = bodies
        self.rc.setPolygonStyle(polygonStyle)
        self.lineWidth = lineWidth

        #                randomSize = 100
        #                variation = mm.v3(randomSize*random.random(), randomSize*random.random(), randomSize*random.random())
        #                resultColor = variation + self.totalColor
        #                for i in range(3):
        #                    if resultColor[i]<0: resultColor[i]=0.
        #                    elif resultColor[i]>255: resultColor[i]=255
        #                glColor3ubv(resultColor)

        #        self.colors = [self.totalColor]*len(self.bodies)
        self.colors = [mm.s2v(self.totalColor)*(1-(float(i)/len(self.bodies) * .5)) for i in range(len(self.bodies))]

    def render(self, renderType=RENDER_OBJECT):
        self.rc.beginDraw()

        for i in range(len(self.bodies)):
            body = self.bodies[i]

            if renderType == RENDER_OBJECT:
                glColor3ubv(self.colors[i])
            else:
                glColor3ubv(self.shadowColor)

            boxSize = body.getShape(); T = body.getFrame()
            glPushMatrix()
            glMultMatrixf(T.T)
            glLineWidth(self.lineWidth)
            self.rc.drawBox(boxSize[0], boxSize[1], boxSize[2])
            glLineWidth(1.)
            glPopMatrix()

class MyFootRenderer(Renderer):
    def __init__(self, boxsize, T, color=(255,0,0), polygonStyle=POLYGON_FILL, lineWidth=1.):
        Renderer.__init__(self, None, color)
        self.boxsize = boxsize
        self.T = T
        self.lineWidth = lineWidth

    def render(self, renderType=RENDER_OBJECT):
        self.rc.beginDraw()
        if renderType == RENDER_OBJECT:
            glColor3ubv(self.totalColor)
        else:
            glColor3ubv(self.shadowColor)

        boxSize = self.boxsize
        glPushMatrix()
        glMultMatrixf(self.T.T)
        glLineWidth(self.lineWidth)
        self.rc.drawBox(boxSize[0], boxSize[1], boxSize[2])
        glLineWidth(1.)
        glPopMatrix()

class BoxesRenderer(Renderer):
    # boxSizes[i] = (lx, ly, lz)
    # Ts[i] = SE3
    def __init__(self, boxSizes, Ts, color=(255,0,0), polygonStyle=POLYGON_FILL, lineWidth=1.):
        Renderer.__init__(self, None, color)
        self.boxSizes = boxSizes
        self.Ts = Ts
        self.rc.setPolygonStyle(polygonStyle)
        self.lineWidth = lineWidth

    def render(self, renderType=RENDER_OBJECT):
        self.rc.beginDraw()
        if renderType == RENDER_OBJECT:
            glColor3ubv(self.totalColor)
        else:
            glColor3ubv(self.shadowColor)
        for i in range(len(self.boxSizes)):
            if (self.boxSizes[i] is not None) and (self.Ts[i] is not None):
                boxSize = self.boxSizes[i]; T = self.Ts[i]
                glPushMatrix()
                glMultMatrixf(T.T)
                glLineWidth(self.lineWidth)
                self.rc.drawBox(boxSize[0], boxSize[1], boxSize[2])
                glLineWidth(1.)
                glPopMatrix()

class CylindersRenderer(Renderer):
    # cylinderSizes[i] = (radius, length_z)
    # Ts[i] = SE3
    def __init__(self, cylinderSizes, Ts, color=(255,0,0)):
        Renderer.__init__(self, None, color)
        self.cylinderSizes = cylinderSizes
        self.Ts = Ts
        self.rc.setNormalStyle(NORMAL_SMOOTH)

    def render(self, renderType=RENDER_OBJECT):
        self.rc.beginDraw()
        if renderType == RENDER_OBJECT:
            glColor3ubv(self.totalColor)
        else:
            glColor3ubv(self.shadowColor)
        for i in range(len(self.cylinderSizes)):
            if self.cylinderSizes[i]!=None and self.Ts[i]!=None:
                cylinderSize = self.cylinderSizes[i]; T = self.Ts[i]
                glPushMatrix()
                glMultMatrixf(T.T)
                self.rc.drawCylinder(cylinderSize[0], cylinderSize[1])
                glPopMatrix()

class SpheresRenderer(Renderer):
    # radius = radius of sphere
    # position = position of center of sphere
    def __init__(self, radiuses, positions, color=(255,0,0), polygonStyle = POLYGON_FILL):
        Renderer.__init__(self, None, color)
        self.radiuses = radiuses
        self.positions = positions
        self.rc.setNormalStyle(NORMAL_SMOOTH)
        self.rc.setPolygonStyle(polygonStyle)
        self.rc.setLineWidth(2.)

    def render(self, renderType=RENDER_OBJECT):
        self.rc.beginDraw()
        glColor3ubv(self.totalColor)
        for i in range(len(self.radiuses)):
            if self.radiuses[i] is not None and self.positions[i] is not None:
                rad = self.radiuses[i]; pos = self.positions[i]

                #                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                eqr=(0.0,-1.0, 0.0, 0.0001)
                glDisable(GL_CULL_FACE)
                glEnable(GL_CLIP_PLANE0)

                #                glEnable(GL_COLOR_MATERIAL);
                #                glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
                #                glColorMaterial(GL_BACK, GL_DIFFUSE);

                glPushMatrix()
                glTranslatef(pos[0], pos[1], pos[2])
                glClipPlane(GL_CLIP_PLANE0, eqr)
                glRotatef(90, 1,0,0)
                self.rc.drawSphere(rad)
                glPopMatrix()

                #                glDisable(GL_COLOR_MATERIAL);

                glDisable(GL_CLIP_PLANE0)
                glEnable(GL_CULL_FACE)


#===============================================================================
# # common class
#===============================================================================
class RenderContext:
    def __init__(self):
        self.quad = gluNewQuadric()
        self.quad2 = gluNewQuadric()
#        gleSetNumSides(12)
        
        self.setPolygonStyle(POLYGON_FILL)
        self.setNormalStyle(NORMAL_SMOOTH)
        self.setLineWidth(1.)
        self.crossLength = .1

        # self.crc = cr.RenderContext()
        
    def __del__(self):
        gluDeleteQuadric(self.quad)
        gluDeleteQuadric(self.quad2)
        
    def setPolygonStyle(self, polygonStyle):
        self.polygonStyle = polygonStyle
        if polygonStyle == POLYGON_LINE:
            gluQuadricDrawStyle(self.quad, GLU_LINE)
            gluQuadricDrawStyle(self.quad2, GLU_LINE)
        elif polygonStyle == POLYGON_FILL:
            gluQuadricDrawStyle(self.quad, GLU_FILL)
            gluQuadricDrawStyle(self.quad2, GLU_FILL)

    def setNormalStyle(self, normalStyle):
        self.normalStyle = normalStyle
        if normalStyle == NORMAL_FLAT:
            gluQuadricDrawStyle(self.quad, GLU_FLAT)
            gluQuadricDrawStyle(self.quad2, GLU_FLAT)
        elif normalStyle == NORMAL_SMOOTH:
            gluQuadricDrawStyle(self.quad, GLU_SMOOTH)
            gluQuadricDrawStyle(self.quad2, GLU_SMOOTH)

    def setLineWidth(self, lineWidth):
        self.lineWidth = lineWidth
            
    def beginDraw(self):
        if self.polygonStyle == POLYGON_LINE:
            glPolygonMode(GL_FRONT, GL_LINE)
        elif self.polygonStyle == POLYGON_FILL:
            glPolygonMode(GL_FRONT, GL_FILL)
        
#        if self.normalStyle == NORMAL_FLAT:
#           gleSetJoinStyle(TUBE_NORM_FACET | TUBE_JN_CAP | TUBE_JN_CUT)
#        elif self.normalStyle == NORMAL_SMOOTH:
#            gleSetJoinStyle(TUBE_NORM_EDGE | TUBE_JN_CAP | TUBE_JN_CUT)
            
        glLineWidth(self.lineWidth)
            
    #===============================================================================
    # draw primitives at origin    
    #===============================================================================
    def drawBox(self, lx, ly, lz):
        glPushMatrix()
        glTranslated(lx/2.,ly/2.,lz/2.)
        glScale(lx, ly, lz)
        #self.crc.drawBox(self.polygonStyle)

        #'''
        if self.polygonStyle == POLYGON_LINE:
            glBegin(GL_LINES)
            glVertex3f(-.5, -.5, -.5)
            glVertex3f(-.5, -.5, +.5)
            glVertex3f(+.5, -.5, -.5)
            glVertex3f(+.5, -.5, +.5)
            glVertex3f(-.5, +.5, -.5)
            glVertex3f(-.5, +.5, +.5)
            glVertex3f(+.5, +.5, -.5)
            glVertex3f(+.5, +.5, +.5)

            glVertex3f(+.5, -.5, -.5)
            glVertex3f(-.5, -.5, -.5)
            glVertex3f(+.5, +.5, -.5)
            glVertex3f(-.5, +.5, -.5)
            glVertex3f(+.5, -.5, +.5)
            glVertex3f(-.5, -.5, +.5)
            glVertex3f(+.5, +.5, +.5)
            glVertex3f(-.5, +.5, +.5)

            glVertex3f(-.5, -.5, -.5)
            glVertex3f(-.5, +.5, -.5)
            glVertex3f(-.5, -.5, +.5)
            glVertex3f(-.5, +.5, +.5)
            glVertex3f(+.5, -.5, +.5)
            glVertex3f(+.5, +.5, +.5)
            glVertex3f(+.5, -.5, -.5)
            glVertex3f(+.5, +.5, -.5)

            glEnd()
        else:
            glBegin(GL_QUADS)
            glNormal3f(0., 0., -1.)
            glVertex3f(-.5, -.5, -.5)
            glNormal3f(0., 0., -1.)
            glVertex3f(-.5, +.5, -.5)
            glNormal3f(0., 0., -1.)
            glVertex3f(+.5, +.5, -.5)
            glNormal3f(0., 0., -1.)
            glVertex3f(+.5, -.5, -.5)

            glNormal3f(0., 0., +1.)
            glVertex3f(-.5, -.5, +.5)
            glNormal3f(0., 0., +1.)
            glVertex3f(+.5, -.5, +.5)
            glNormal3f(0., 0., +1.)
            glVertex3f(+.5, +.5, +.5)
            glNormal3f(0., 0., +1.)
            glVertex3f(-.5, +.5, +.5)

            glNormal3f(-1., 0., 0.)
            glVertex3f(-.5, -.5, -.5)
            glNormal3f(-1., 0., 0.)
            glVertex3f(-.5, -.5, +.5)
            glNormal3f(-1., 0., 0.)
            glVertex3f(-.5, +.5, +.5)
            glNormal3f(-1., 0., 0.)
            glVertex3f(-.5, +.5, -.5)

            glNormal3f(+1., 0., 0.)
            glVertex3f(+.5, -.5, -.5)
            glNormal3f(+1., 0., 0.)
            glVertex3f(+.5, +.5, -.5)
            glNormal3f(+1., 0., 0.)
            glVertex3f(+.5, +.5, +.5)
            glNormal3f(+1., 0., 0.)
            glVertex3f(+.5, -.5, +.5)

            glNormal3f(0., -1., 0.)
            glVertex3f(-.5, -.5, -.5)
            glNormal3f(0., -1., 0.)
            glVertex3f(+.5, -.5, -.5)
            glNormal3f(0., -1., 0.)
            glVertex3f(+.5, -.5, +.5)
            glNormal3f(0., -1., 0.)
            glVertex3f(-.5, -.5, +.5)

            glNormal3f(0., +1., 0.)
            glVertex3f(-.5, +.5, -.5)
            glNormal3f(0., +1., 0.)
            glVertex3f(-.5, +.5, +.5)
            glNormal3f(0., +1., 0.)
            glVertex3f(+.5, +.5, +.5)
            glNormal3f(0., +1., 0.)
            glVertex3f(+.5, +.5, -.5)
            glEnd()
        #'''
        glPopMatrix()

    def drawCenteredBox(self, lx, ly, lz):
        glPushMatrix()
        glScale(lx, ly, lz)
        self.crc.drawBox(self.polygonStyle)
        glPopMatrix()

    def drawCylinder(self, radius, length_z):
        gluCylinder(self.quad, radius, radius, length_z, 16, 1)

    def drawCapsule(self, radius, length_z):
        _SLICE_SIZE = CAPSULE_SLICE_SIZE
        glPushMatrix()
        glTranslatef(0., 0., -length_z/2.)
        gluSphere(self.quad2, radius, _SLICE_SIZE, _SLICE_SIZE)

        glTranslatef(0., 0., +length_z)
        gluSphere(self.quad2, radius, _SLICE_SIZE, _SLICE_SIZE)

        glTranslatef(0., 0., -length_z)

        gluCylinder(self.quad, radius, radius, length_z, _SLICE_SIZE, 1)

        glPopMatrix()

    def drawSphere(self, radius):
        _SLICE_SIZE = SPHERE_SLICE_SIZE
        gluSphere(self.quad2, radius, _SLICE_SIZE, _SLICE_SIZE)
        # SLICE = 20; STACK = 20
        # if self.polygonStyle == POLYGON_LINE:
        #     glutWireSphere(radius, SLICE, STACK)
        # else:
        #     glutSolidSphere(radius, SLICE, STACK)

    def drawCircle(self, radius):
        _SLICE_SIZE = 32
        glBegin(GL_LINE_LOOP)
        for i in range(_SLICE_SIZE):
            glVertex2f(radius * math.cos(2.*math.pi*float(i)/_SLICE_SIZE), radius * math.sin(2.*math.pi*float(i)/_SLICE_SIZE))
        glEnd()

    def drawCircleFilled(self, radius):
        _SLICE_SIZE = 32
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(0., 0.)
        for i in range(_SLICE_SIZE+1):
            glVertex2f(radius * math.cos(2.*math.pi*float(i)/_SLICE_SIZE), radius * math.sin(2.*math.pi*float(i)/_SLICE_SIZE))
        glEnd()

    def drawCapsule2DFilled(self, radius, height):
        _SLICE_SIZE = 16
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(0., 0.)
        r, h = radius, height

        for i in range(_SLICE_SIZE):
            glVertex2f(r*math.cos(math.pi*float(i)/_SLICE_SIZE), h/2. + r*math.sin(math.pi*float(i)/_SLICE_SIZE))
        for i in range(_SLICE_SIZE):
            glVertex2f(-r*math.cos(math.pi*float(i)/_SLICE_SIZE), -h/2. - r*math.sin(math.pi*float(i)/_SLICE_SIZE))

        glVertex2f(r, h/2.)
        glEnd()

    def drawCapsule2D(self, radius, height):
        _SLICE_SIZE = 16
        # glLineWidth(1.)
        glBegin(GL_LINE_LOOP)
        r, h = radius, height

        for i in range(_SLICE_SIZE):
            glVertex2f(r*math.cos(math.pi*float(i)/_SLICE_SIZE), h/2. + r*math.sin(math.pi*float(i)/_SLICE_SIZE))
        for i in range(_SLICE_SIZE):
            glVertex2f(-r*math.cos(math.pi*float(i)/_SLICE_SIZE), -h/2. - r*math.sin(math.pi*float(i)/_SLICE_SIZE))

        glVertex2f(r, h/2.)
        glEnd()

    #===============================================================================
    # draw primitives at its position        
    #===============================================================================
    def drawPoint(self, point):
        glPointSize(3.0)
        glBegin(GL_POINTS)
        glVertex3fv(point)
        glEnd()
        
    def drawCross(self, point):
        glPushMatrix()
        glTranslatef(point[0], point[1], point[2])
        glBegin(GL_LINES)
        crossLength = self.crossLength
        glVertex3f(crossLength/2.,0,0)    # x
        glVertex3f(-crossLength/2.,0,0)
        glVertex3f(0,crossLength/2.,0)    # y
        glVertex3f(0,-crossLength/2.,0)
        glVertex3f(0,0,crossLength/2.)    # z
        glVertex3f(0,0,-crossLength/2.)
        glEnd()
        glPopMatrix()
        
    def drawCube(self, point):
        ygh.beginDraw()
        ygh.drawPoint(point)
        ygh.endDraw()
        glPushMatrix()
        glTranslated(point[0], point[1], point[2])
        glutWireCube(.1)
        glPopMatrix()
            
    def drawLine(self, startPos, endPos):
        glBegin(GL_LINES)
#        glVertex3fv(startPos)
#        glVertex3fv(endPos)
        glVertex3f(startPos[0], startPos[1], startPos[2])
        glVertex3f(endPos[0], endPos[1], endPos[2])
        glEnd()
        
    def draw2DArrow(self, startPos, endPos, vector=None, lineWidth=.02):
        if vector is None:
            vector = [endPos[i]-startPos[i] for i in range(3)]
        elif startPos is None:
            startPos = [endPos[i]-vector[i] for i in range(3)]
        
#        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glDisable(GL_CULL_FACE)
        glPushMatrix()
        
        length = mm.length(vector)
        arrowT = mm.Rp2T(mm.getSO3FromVectors((length,0,0), vector), startPos)
        glMultMatrixf(arrowT.transpose())
        
        triWidth = lineWidth * 3
        triLength = triWidth * 1.2
        
        angles = [0, 90]
        for angle in angles:
            glRotatef(angle, 1,0,0)
        
            # line part
            glBegin(GL_QUADS)        
            glVertex3f(0,0,lineWidth/2)
            glVertex3f(0,0,-lineWidth/2)
            glVertex3f(length - triLength,0,-lineWidth/2)
            glVertex3f(length - triLength,0,+lineWidth/2)
            glEnd()
            
            # triangle part
            glBegin(GL_TRIANGLES)
            glVertex3f(length - triLength, 0, triWidth/2)
            glVertex3f(length - triLength, 0, -triWidth/2)
            glVertex3f(length, 0, 0)
            glEnd()
    
        glPopMatrix()
        glEnable(GL_CULL_FACE)
        
    def drawArrow(self, startPos, endPos, vector=None, lineWidth=.02):
        if vector is None:
            vector = [endPos[i]-startPos[i] for i in range(3)]
        elif startPos is None:
            startPos = [endPos[i]-vector[i] for i in range(3)]

        length = mm.length(vector)
        if length==0.: return

        glPushMatrix()
        
        arrowT = mm.Rp2T(mm.getSO3FromVectors((length,0,0), vector), startPos)
        glMultMatrixf(arrowT.transpose())

        # triWidth = lineWidth * 3
        # triLength = triWidth * 1.2
        triWidth = lineWidth * 1.5
        triLength = triWidth * 1.2

        # line + cone all parts
        glePolyCone(((0,0,0), (0,0,0), (length-triLength,0,0), (length-triLength,0,0), (length,0,0), (length,0,0)), None,
                   (lineWidth/2., lineWidth/2., lineWidth/2., triWidth/2., 0, 0))
        
        glPopMatrix()

    def drawJointArrow(self, startPos, endPos, vector=None, lineWidth=.02):
        if vector is None:
            vector = [endPos[i]-startPos[i] for i in range(3)]
        elif startPos is None:
            startPos = [endPos[i]-vector[i] for i in range(3)]

        length = mm.length(vector)
        if length==0.: return

        glPushMatrix()

        arrowT = mm.Rp2T(mm.getSO3FromVectors((length,0,0), vector), startPos)
        glMultMatrixf(arrowT.transpose())

        # triWidth = lineWidth * 3
        # triLength = triWidth * 1.2
        triWidth = lineWidth * 1.5
        # triLength = triWidth * 1.2
        triLength = length

        # line + cone all parts
        glePolyCone(((0,0,0), (0,0,0), (length-triLength,0,0), (length-triLength,0,0), (length,0,0), (length,0,0)), None,
                    (lineWidth/2., lineWidth/2., lineWidth/2., triWidth/2., 0, 0))

        glPopMatrix()

    def drawCircularArrow(self, startPos, endPos, rotVec=None, lineWidth=.02, radius=.1):
        if rotVec is None:
            rotVec = [endPos[i]-startPos[i] for i in range(3)]
        elif startPos is None:
            startPos = [endPos[i]-rotVec[i] for i in range(3)]
        
        length = mm.length(rotVec)
        if length==0.: return

        glPushMatrix()
        
        axisT = mm.Rp2T(mm.getSO3FromVectors((0,0,length), rotVec), startPos)
        glMultMatrixf(axisT.transpose())
        
        triWidth = lineWidth * 3
        triLength = triWidth * 1.2
        
        # axis
#        self.drawLine((0,0,0), (0,0,length))
#        glePolyCylinder(((0,0,0), (0,0,0), (0,0,length), (0,0,length)), None, lineWidth/4.)
        
        # circular line part
#        gleHelicoid( rToroid , startRadius , drdTheta , startZ , dzdTheta , 
#                     startXform , dXformdTheta , startTheta , sweepTheta )
        sweepTheta = 2*math.pi*length*mm.DEG
#        gleHelicoid( lineWidth/2., radius,       0.,        0.,    radius,
#                        None,         None,            0.,     sweepTheta)
        
        # cone part
        glPushMatrix()
        glRotatef(sweepTheta, 0,0,1)
        glTranslatef(radius, 0, radius * (sweepTheta/360.))
        glRotatef(-90, 1,0,0)
#        glePolyCone(((0,0,0), (0,0,0), (0,0,triLength), (0,0,triLength)), None, 
#                    (triWidth/2., triWidth/2., 0, 0))
        glPopMatrix()
        
        glPopMatrix()

    '''
    def renderSelectedOdeGeom(self, geom, color):
        if type(geom) == ode.GeomBox:
            lx, ly, lz = geom.getLengths()
            x,y,z = geom.getPosition()
            R = geom.getRotation()
            se3 = [R[0], R[3], R[6], 0.,
                   R[1], R[4], R[7], 0.,
                   R[2], R[5], R[8], 0.,
                   x, y, z, 1.0]
            glPushMatrix()
            glMultMatrixd(se3)
            ygh.drawCoordinate(color)
            glScaled(1.1,1.1,1.1)
            glTranslated(-lx/2.,-ly/2.,-lz/2.)
            self.drawBox(lx, ly, lz)
            glPopMatrix()
            
    def renderOdeGeom(self, geom):
        if type(geom) == ode.GeomBox:
            lx, ly, lz = geom.getLengths()
            x,y,z = geom.getPosition()
            R = geom.getRotation()
            se3 = [R[0], R[3], R[6], 0.,
                   R[1], R[4], R[7], 0.,
                   R[2], R[5], R[8], 0.,
                   x, y, z, 1.0]
            glPushMatrix()
            glMultMatrixd(se3)
            glTranslated(-lx/2.,-ly/2.,-lz/2.)
            self.drawBox(lx, ly, lz)
            glPopMatrix()
            
        elif type(geom) == ode.GeomCapsule:
            radius, length_z = geom.getParams()
            
            x,y,z = geom.getPosition()
            R = geom.getRotation()
            se3 = [R[0], R[3], R[6], 0.,
                   R[1], R[4], R[7], 0.,
                   R[2], R[5], R[8], 0.,
                   x, y, z, 1.0]
            glPushMatrix()
            glMultMatrixd(se3)
            glTranslated(0,0,-length_z/2.)
            self.drawCylinder(radius, length_z)
            glPopMatrix()
    
        elif type(geom) == ode.GeomSphere:
            radius = geom.getRadius()
        
        elif type(geom) == ode.GeomPlane:
            (a, b, c), d = geom.getParams()
            glPushMatrix()
            glTranslatef(0,d,0)
            glScale(10,0,10)
            glutWireCube(1)
            glPopMatrix()
            
        elif type(geom) == ode.GeomRay:
            length = geom.getLength()
    '''

if __name__=='__main__':
    from fltk import *
    import math
    import Resource.ysOgreDataLoader as yol
    import GUI.ysSimpleViewer as ysv
    import Resource.ysMotionLoader as yf

    def test_MeshRenderer():
        meshFilePath = '../samples/woody2_15.mesh.xml'
        mesh = yol.readOgreMeshFileAsMesh(meshFilePath)

        viewer = ysv.SimpleViewer()
        viewer.doc.addRenderer('mesh', MeshRenderer(mesh))
        viewer.doc.addObject('mesh', mesh)

        viewer.startTimer(1./30.)
        viewer.show()

        Fl.run()

    def test_FramesRenderer_OrientationsRenderer():
        frame0 = mm.I_SE3()
        frame1 = mm.Rp2T(mm.exp(mm.v3(0,1,0), math.pi/8.), (1,0,0))

        viewer = ysv.SimpleViewer()
        viewer.doc.addRenderer('frame0', FramesRenderer([frame0], (255,0,0)))
        viewer.doc.addRenderer('frame1', FramesRenderer([frame1], (255,0,0)))
        viewer.doc.addRenderer('orientation0', OrientationsRenderer([mm.T2R(frame0)], [mm.T2p(frame0)], (0,255,0)))
        viewer.doc.addRenderer('orientation1', OrientationsRenderer([mm.T2R(frame1)], [mm.T2p(frame1)], (0,255,0)))

        viewer.show()
        Fl.run()

    def test_ForcesRenderer_TorquesRenderer():
        forces = [None]*5
        points1 = [None]*5
        torques = [None]*5
        points2 = [None]*5

        for i in range(len(forces)):
            forces[i] = (0,i,0)
            points1[i] = (i,0,0)

        for i in range(len(torques)):
            points2[i] = (-i,0,0)
            torques[i] = (0,0,i)

        viewer = ysv.SimpleViewer()
        viewer.doc.addRenderer('forces', ForcesRenderer(forces, points1, (255,0,0), 1., .1))
        viewer.doc.addRenderer('torques', TorquesRenderer(torques, points2, (0,255,0), 1., .1, .3))

        viewer.show()
        Fl.run()

    def test_primitives_renderers():
        boxSize = [1,1,1]
        boxFrame = mm.I_SE3()

        cylinderSize = [1,1]
        cylinderFrame = mm.I_SE3()

        sphereRadius = 1.
        spherePosition = (0,0.5,0)

        arrowVector = (1,0,0)
        arrowPoint = (0,0,0)

        viewer = ysv.SimpleViewer()
        # viewer.doc.addRenderer('box', BoxesRenderer([boxSize], [boxFrame], (255,0,0)))
        # viewer.doc.addRenderer('cylinder', CylindersRenderer([cylinderSize], [cylinderFrame], (0,0,255)))
        # viewer.doc.addRenderer('sphere', SpheresRenderer([sphereRadius], [spherePosition], (0,0,255)))
        viewer.doc.addRenderer('sphere', WideArrowRenderer([arrowVector], [arrowPoint], (255,0,0), 1., .1))

        viewer.show()
        Fl.run()


    pass
    test_MeshRenderer()
    #    test_FramesRenderer_OrientationsRenderer()
    #    test_ForcesRenderer_TorquesRenderer()
    # test_primitives_renderers()
