# +-------------------------------------------------------------------------
# | ysGlHelper.py
# |
# | Author: Yoonsang Lee
# +-------------------------------------------------------------------------
# | COPYRIGHT:
# |    Copyright Yoonsang Lee 2013
# |    See the included COPYRIGHT.txt file for further details.
# |    
# |    This file is part of the DataDrivenBipedController.
# |    DataDrivenBipedController is free software: you can redistribute it and/or modify
# |    it under the terms of the MIT License.
# |
# |    You should have received a copy of the MIT License
# |    along with DataDrivenBipedController.  If not, see <mit-license.org>.
# +-------------------------------------------------------------------------

import numpy
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from PyCommon.modules.Math import mmMath as mmMath

class DrawingSet:
    def __init__(self, default_color = (0,255,0)):
        self.pointMap = {}
        self.vectorMap = {}
        self.vectorOriginMap = {}
        self.SO3Map = {}
        self.SO3OriginMap = {}
        self.colorMap = {}
        
        self.coordinate = mmMath.I_SE3()
        
        self.point_size = 3.0
        self.line_width = 1.0
        self.default_color = default_color
        
    def addPoint(self, name, point, color = None):
        self.pointMap[name] = point
        if not color:
            color = self.default_color
        self.colorMap[name] = color
        
    def addVector(self, name, vector, origin = (0,0,0), color = None):
        self.vectorMap[name] = vector
        self.vectorOriginMap[name] = origin
        if not color:
            color = self.default_color
        self.colorMap[name] = color
        
    def addSO3(self, name, SO3, origin = (0,0,0), color = None):
        self.SO3Map[name] = SO3
        self.SO3OriginMap[name] = origin
        if not color:
            color = self.default_color
        self.colorMap[name] = color
        
    def beginDraw(self):
        beginDraw()
        glMultMatrixf(self.coordinate.transpose())

    def drawAll(self):
        self.drawPoints()
        self.drawVectors()
        self.drawSO3s()
        self.drawCoordinate()
    def drawPoints(self):
        for name in self.pointMap.keys():
            self.drawPoint(name)
    def drawVectors(self):
        for name in self.vectorMap.keys():
            self.drawVector(name)
    def drawSO3s(self):
        for name in self.SO3Map.keys():
            self.drawSO3(name)
    def drawPoint(self, name):
        drawPoint(self.pointMap[name], self.colorMap.get(name,self.default_color), self.point_size, name)
    def drawVector(self, name):
        drawVector(self.vectorMap[name], self.vectorOriginMap.get(name,(0,0,0)), self.colorMap.get(name,self.default_color), self.line_width, name)
    def drawSO3(self, name):
        drawSO3(self.SO3Map[name], self.SO3OriginMap.get(name,(0,0,0)), self.colorMap.get(name,self.default_color), self.line_width, name)
    def drawCoordinate(self):
        drawCoordinate(self.default_color, .5, 1.0)
        
    def endDraw(self):
        endDraw()

    def __str__(self):
        string = ''
        string += '# pointMap\n'
        for name, point in self.pointMap.items():
            string += name + str(point)+'\n'
        string += '# vectorMap\n'
        for name, vector in self.vectorMap.items():
            string += name + str(vector) + 'origin -' + str(self.vectorOriginMap.get(name,(0,0,0)))+'\n'
        string += '# SO3Map\n'
        for name, SO3 in self.SO3Map.items():
            print(name, SO3)
        string += '# coordinate\n'+str(self.coordinate)
        return string


def beginDraw():
    glPushMatrix()
    glPushAttrib(GL_CURRENT_BIT|GL_POINT_BIT|GL_LINE_BIT|GL_LIGHTING_BIT|GL_ENABLE_BIT)
    glShadeModel(GL_SMOOTH)

def drawPolygon(vertices, color=(0,255,0), name=''):
    glColor3ubv(color)
    glDisable(GL_CULL_FACE)
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_LINE)
    glBegin(GL_TRIANGLES)
    for v in vertices:
        glVertex3fv(v)
    glEnd()

def drawPoint(point, color=(0,255,0), size=3.0, name=''):
    glColor3ubv(color)
    glPointSize(size)
    glBegin(GL_POINTS)
    glVertex3fv(point)
    glEnd()
   
def drawVector(vector, origin=numpy.array([0,0,0]), color=(0,255,0), lineWidth=1.0, name=''):
    glLineWidth(lineWidth)
    glBegin(GL_LINES)
    glColor3ubv((255,255,255))
    glVertex3fv(origin)
    glColor3ubv(color)
    glVertex3fv((origin[0]+vector[0],origin[1]+vector[1],origin[2]+vector[2]))
    glEnd()
    
def drawLine(startPos, endPos, color=(0,255,0), lineWidth=1.0, name=''):
    glLineWidth(lineWidth)
    glBegin(GL_LINES)
    glColor3ubv(color)
    glVertex3fv(startPos)
    glVertex3fv(endPos)
    glEnd()
    
def drawSO3(SO3, origin=numpy.array([0,0,0]), color=(0,255,0), lineWidth=1.0, name=''):
    glEnable(GL_LINE_STIPPLE)
    glLineStipple(2, 0xFAFA)
    drawVector(mmMath.logSO3(SO3), origin, color, lineWidth)
    glDisable(GL_LINE_STIPPLE)
    
def drawCoordinate(color=(0,255,0), axisLength = .5, lineWidth=1.0):
    glLineWidth(lineWidth)
    glColor3ubv(color)
    glBegin(GL_LINES)
    fontSize = axisLength/20
    glVertex3f(axisLength,0,0)    # x
    glVertex3f(0,0,0)
    glVertex3f(axisLength-fontSize,fontSize,0)
    glVertex3f(axisLength+fontSize,-fontSize,0)
    glVertex3f(axisLength+fontSize,+fontSize,0)
    glVertex3f(axisLength-fontSize,-fontSize,0)
    glVertex3f(0,axisLength,0)    # y
    glVertex3f(0,0,0)
    glVertex3f(fontSize,axisLength+fontSize,0)
    glVertex3f(0,axisLength,0)
    glVertex3f(-fontSize,axisLength+fontSize,0)
    glVertex3f(0,axisLength,0)
    glVertex3f(0,0,axisLength)    # z
    glVertex3f(0,0,0)
    glEnd()
    
def drawCross(point, color=(0,255,0), crossLength=.1, lineWidth=1.0, name=''):
    glLineWidth(lineWidth)
    glColor3ubv(color)
    glPushMatrix()
    glTranslatef(point[0], point[1], point[2])
    glBegin(GL_LINES)
    glVertex3f(crossLength/2.,0,0)    # x
    glVertex3f(-crossLength/2.,0,0)
    glVertex3f(0,crossLength/2.,0)    # y
    glVertex3f(0,-crossLength/2.,0)
    glVertex3f(0,0,crossLength/2.)    # z
    glVertex3f(0,0,-crossLength/2.)
    glEnd()
    glPopMatrix()
   
def endDraw():
    glPopAttrib()
    glPopMatrix()    