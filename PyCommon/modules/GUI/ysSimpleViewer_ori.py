from fltk import *

from PyCommon.modules.Motion import ysMotion as ym
from PyCommon.modules.GUI import ysBaseUI as ybu
from PyCommon.modules.GUI import ysViewer3 as yv3
from PyCommon.modules.GUI import tree

# EVENTS
EV_addRenderer           = 0
EV_setRendererVisible    = 1
EV_addObject             = 2
EV_selectObjectElement   = 3
EV_selectObject          = 4

class SimpleSetting(ybu.BaseSettings):
    def __init__(self, x=100, y=100, w=1200, h=900):
        ybu.BaseSettings.__init__(self, x, y, w, h)
        self.camera = yv3.Camera().__dict__
        self.ortho = False
        self.viewMode = yv3.VIEW_PERSPECTIVE
        self.prevRotX = 0
        self.prevRotY = 0
#        self.infoWndIdx = 0
    def setToApp(self, window):
        ybu.BaseSettings.setToApp(self, window)
        window.motionViewWnd.glWindow.camera.__dict__ = self.camera
        window.motionViewWnd.glWindow.projectionOrtho = self.ortho
        window.motionViewWnd.glWindow.viewMode = self.viewMode
        window.motionViewWnd.glWindow.prevRotX = self.prevRotX
        window.motionViewWnd.glWindow.prevRotY = self.prevRotY
#        window.objectInfoWnd.currentChildWndIdx = self.infoWndIdx
    def getFromApp(self, window):
        ybu.BaseSettings.getFromApp(self, window)
        self.camera = window.motionViewWnd.glWindow.camera.__dict__
        self.ortho = window.motionViewWnd.glWindow.projectionOrtho
        self.viewMode = window.motionViewWnd.glWindow.viewMode
        self.prevRotX = window.motionViewWnd.glWindow.prevRotX
        self.prevRotY = window.motionViewWnd.glWindow.prevRotY
#        self.infoWndIdx = window.objectInfoWnd.currentChildWndIdx

class SimpleViewer(ybu.BaseWnd):
    def __init__(self, rect=None, title='SimpleViewer'):
        ybu.BaseWnd.__init__(self, rect, title, SimpleSetting())
        self.doc = SimpleDoc()
        self.begin()
        panelWidth = 180
        self.motionViewWnd = MotionViewWnd(0, 0, self.w()-panelWidth, self.h(), self.doc)
        t = .3
        self.renderersWnd = RenderersWnd(self.w()-panelWidth, 0, panelWidth, int(self.h()*t), self.doc)
        self.objectInfoWnd = ObjectInfoWnd(self.w()-panelWidth, int(self.h()*t), panelWidth, int(self.h()*(1-t)), self.doc)
        self.end()
        self.resizable(self.motionViewWnd)
        self.size_range(600, 400)
    def startTimer(self, timeInterval):
        self.motionViewWnd.startTimer(timeInterval)
    def endTimer(self):
        self.motionViewWnd.endTimer()
    def setTimeInterval(self, timeInterval):
        self.motionViewWnd.setTimeInterval(timeInterval)
    def show(self):
        ybu.BaseWnd.show(self)
        self.motionViewWnd.show()
    def setPreFrameCallback(self, callback):
        self.motionViewWnd.preFrameCallback = callback
    def setPreFrameCallback_Always(self, callback):
        self.motionViewWnd.preFrameCallback_Always = callback
    def setSimulateCallback(self, callback):
        self.motionViewWnd.simulateCallback = callback
    def setPostFrameCallback(self, callback):
        self.motionViewWnd.postFrameCallback = callback
    def setPostFrameCallback_Always(self, callback):
        self.motionViewWnd.postFrameCallback_Always = callback
    def setExtraDrawCallback(self, callback):
        self.motionViewWnd.glWindow.extraDrawCallback = callback
#    def setRecSimulObjs(self, objs):
#        self.motionViewWnd.setRecSimulObjs(objs)
    def getMaxFrame(self):
        return self.motionViewWnd.getMaxFrame()
    def setMaxFrame(self, maxFrame):
        self.motionViewWnd.setMaxFrame(maxFrame)
    def record(self, bRec):
        self.motionViewWnd.record(bRec)
    def play(self):
        self.motionViewWnd.play()
    def setCurrentFrame(self, frame):
        self.motionViewWnd.setCurrentFrame(frame)
    def getCurrentFrame(self):
        return self.motionViewWnd.getCurrentFrame()
    def setCameraTarget(self, targetPos):
        self.motionViewWnd.glWindow.camera.center[0] = targetPos[0]
        self.motionViewWnd.glWindow.camera.center[2] = targetPos[2]
    def initialize(self):
        self.doc.initialize()
        self.motionViewWnd.initialize()
        
class SimpleDoc(ybu.Subject):
    def __init__(self):
        ybu.Subject.__init__(self)
        
        self.rendererNames = []
        self.rendererMap = {}
        self.renderersVisible = {}

        self.motionNames = []
        self.motionMap = {}
        self.motionSystem = ym.MotionSystem()
        
        self.objectNames = []
        self.objectMap = {}
        self.selectedObject = None
    def initialize(self):
        self.removeAllRenderers()
        self.removeAllObjects()
    def removeAllRenderers(self):
        del self.rendererNames[:]
        self.rendererMap.clear()
        self.renderersVisible.clear()
        self.notify(EV_addRenderer)
    def removeAllObjects(self):
        del self.objectNames[:]
        self.objectMap.clear()
        self.motionSystem.removeAllMotions()
    def addRenderer(self, name, renderer, visible=True):
        self.rendererNames.append(name)
        self.rendererMap[name] = renderer
        self.renderersVisible[name] = visible
        self.notify(EV_addRenderer)
    def setRendererVisible(self, name, visible):
        self.renderersVisible[name] = visible
        self.notify(EV_setRendererVisible)
    def getVisibleRenderers(self):
        ls = []
        for name, renderer in self.rendererMap.items():
            if self.renderersVisible[name]:
                ls.append(renderer)
        return ls
    def getInvisibleRenderers(self):
        ls = []
        for name, renderer in self.rendererMap.items():
            if not self.renderersVisible[name]:
                ls.append(renderer)
        return ls
    def addObject(self, name, object):
        self.objectNames.append(name)
        self.objectMap[name] = object
        if isinstance(object, ym.Motion):
            self.motionSystem.addMotion(object)
        self.notify(EV_addObject)
    def selectObjectElement(self, element):
        for renderer in self.rendererMap.values():
            renderer.selectedElement = element 
        self.notify(EV_selectObjectElement)
    def selectObject(self, objectName):
        self.selectedObject = self.objectMap[objectName]
        self.notify(EV_selectObject)
    
class MotionViewWnd(yv3.MotionViewer, ybu.Observer):
    def __init__(self, x, y, w, h, doc):
        yv3.MotionViewer.__init__(self, x, y, w, h)
        self.doc = doc
        self.doc.attach(self)
    def update(self, ev, doc):
        if ev==EV_addRenderer or ev==EV_setRendererVisible:
            self.setRenderers(doc.getVisibleRenderers())
            self.setInvisibleRendrers(doc.getInvisibleRenderers())
        elif ev==EV_addObject:
            self.setMotionSystem(doc.motionSystem)
            self.setStateObjects(doc.objectMap.values())
        self.glWindow.redraw()

class RenderersWnd(Fl_Window, ybu.Observer):
    def __init__(self, x, y, w, h, doc):
        Fl_Window.__init__(self, x, y, w, h)
        self.doc = doc
        self.doc.attach(self)
        self.box(FL_PLASTIC_UP_BOX)
        self.begin()
        self.rx = 5; self.ry = 5; self.rw = w-10; self.rh = h-10
        self.renderersChk = Fl_Check_Browser(self.rx, self.ry, self.rw, self.rh, '')
        self.renderersChk.type(FL_MULTI_BROWSER)
#        self.renderersChk.callback(self.onClickBrowser)
        self.end()

    def update(self, ev, doc):
        if ev == EV_addRenderer or ev == EV_setRendererVisible:
            self.renderersChk.clear()
            for name in doc.rendererNames:
                self.renderersChk.add(name, doc.renderersVisible[name])

    def onClickBrowser(self, x, y):
        i = (y-2)//16
        if 0 <= i < self.renderersChk.nitems():
            self.doc.setRendererVisible(self.renderersChk.text(i+1), not self.renderersChk.checked(i+1))

    def handle(self, event):
        if event == FL_PUSH:
            x = Fl.event_x()
            y = Fl.event_y()
            if (self.rx <= x <= self.rx+self.rw) and (self.ry <= y <= self.ry+self.rh):
                self.onClickBrowser(x-self.rx, y-self.ry)
        return Fl_Window.handle(self, event)
    
    def resize(self, x, y, w, h):
        self.renderersChk.size(self.renderersChk.w(), h-10)
        Fl_Window.resize(self, x, y, w, h)


class ObjectInfoWnd(Fl_Window, ybu.Observer):
    def __init__(self, x, y, w, h, doc):
        Fl_Window.__init__(self, x, y, w, h)
        self.doc = doc
        self.doc.attach(self)
#        self.box(FL_PLASTIC_UP_BOX)
        self.begin()
        self.objectNames = Fl_Choice(5,5,w-10,20,'')
        self.objectNames.align(FL_ALIGN_LEFT)
        self.objectNames.value(0)
        self.objectNames.callback(self.onChangeObjectName)
        self.motionSkeletonWnd = MotionSkeletonWnd(0, 25, self.w(), self.h()-25, self.doc)
        self.meshInfoWnd = MeshInfoWnd(0, 25, self.w(), self.h()-25, self.doc)
        self.end()
        #self.childWnds = [self.motionSkeletonWnd]
        self.childWnds = [self.motionSkeletonWnd, self.meshInfoWnd]
        for cw in self.childWnds:
            cw.hide()
        self.currentChildWndIdx = 0
        self.resizable(self.motionSkeletonWnd)
    def update(self, ev, doc):
        if ev==EV_addObject:
            self.objectNames.clear()
            for objectName in doc.objectNames:
                idx = self.objectNames.add(objectName)
        elif ev==EV_selectObject:
            if isinstance(self.doc.selectedObject, ym.Motion):
                self.currentChildWndIdx = 0
            for i in range(len(self.childWnds)):
                if i == self.currentChildWndIdx:
                    self.childWnds[i].show()
                else:
                    self.childWnds[i].hide()
    def onChangeObjectName(self, ptr):
        self.doc.selectObject(ptr.text(ptr.value()))
        self.doc.notify(EV_selectObject)


class MotionSkeletonWnd(Fl_Window, ybu.Observer):
    def __init__(self, x, y, w, h, doc):
        Fl_Window.__init__(self, x, y, w, h)
        self.doc = doc
        self.doc.attach(self)
#        self.box(FL_PLASTIC_UP_BOX)        
        self.begin()
        self.tree= tree.Fl_Tree(5,5,w-10,h-10,'')
        self.tree.align(FL_ALIGN_TOP)
        self.tree.on_select = self.onSelectTree
        self.end()
        self.resizable(self.tree)
    def update(self, ev, doc):
        if ev==EV_selectObject:
            if isinstance(self.doc.selectedObject, ym.Motion):
                self.selectMotion(self.doc.selectedObject)
    def selectMotion(self, motion):
        self.tree.clearTree()
        posture = motion[0]
        if isinstance(posture, ym.JointPosture):
            rootJoint = posture.skeleton.root
            rootNode = self.tree.append(rootJoint.name, rootJoint)
            rootNode.open()
            self._buildJointTree(rootJoint, rootNode)
    def _buildJointTree(self, parentJoint, parentNode):
        for joint in parentJoint.children:
            node = parentNode.append(joint.name, joint)
            node.open()
            self._buildJointTree(joint, node)
    def onSelectTree(self, node):
        self.doc.selectObjectElement(node.payload)


class MeshInfoWnd(Fl_Window, ybu.Observer):
    def __init__(self, x, y, w, h, doc):
        Fl_Window.__init__(self, x, y, w, h)
        self.doc = doc
        self.doc.attach(self)
        #        self.box(FL_PLASTIC_UP_BOX)
        self.begin()
        gap = 5
        lh = 15     # label height
        #        self.brwVertices = Fl_Select_Browser(gap, lh+gap, w-gap*2, int(h/2.)-lh-gap-gap, 'Vertices')
        self.brwVertices = Fl_Hold_Browser(gap, lh+gap, int(w/2)-gap*2+int(gap/2.), h-lh-gap-gap, 'Vertices')
        self.brwVertices.align(FL_ALIGN_LEFT_TOP)
        self.brwVertices.callback(self.onSelectBrwVertices)
        #        self.brwFaces = Fl_Select_Browser(gap, int(h/2.)+lh, w-gap*2, int(h/2.)-lh-gap, 'Faces')
        self.brwFaces = Fl_Hold_Browser(int(w/2.)+int(gap/2.), lh+gap, int(w/2)-gap*2+int(gap/2.), h-lh-gap-gap, 'Faces')
        self.brwFaces.align(FL_ALIGN_LEFT_TOP)
        self.brwFaces.callback(self.onSelectBrwFaces)
        self.end()
    def resize(self, x, y, w, h):
        gap = 5
        lh = 15     # label height
        #        self.brwVertices.size(w-gap*2, int(h/2.)-lh-gap-gap)
        #        self.brwFaces.resize(gap, int(h/2.)+lh, w-gap*2, int(h/2.)-lh-gap)
        self.brwVertices.resize(gap, lh+gap, int(w/2)-gap*2+int(gap/2.), h-lh-gap-gap)
        self.brwFaces.resize(int(w/2.)+int(gap/2.), lh+gap, int(w/2)-gap*2+int(gap/2.), h-lh-gap-gap)
        Fl_Window.resize(self, x, y, w, h)
    def update(self, ev, doc):
        if ev==EV_selectObject:
            pass
    def selectMesh(self, mesh):
        self.brwVertices.label('Vertices:%d'%len(mesh.vertices))
        self.brwVertices.clear()
        for i in range(len(mesh.vertices)):
            self.brwVertices.add(str(i), mesh.vertices[i])
        self.brwFaces.label('Faces:%d'%len(mesh.faces))
        self.brwFaces.clear()
        for i in range(len(mesh.faces)):
            self.brwFaces.add(str(i), mesh.faces[i])
    def onSelectBrwFaces(self, ptr):
        self.doc.selectObjectElement(ptr.data(ptr.value()))
    def onSelectBrwVertices(self, ptr):
        self.doc.selectObjectElement(ptr.data(ptr.value()))

if __name__=='__main__':
    import time
    import Resource.ysMotionLoader as yf
    import Renderer.ysRenderer as yr
    import Resource.ysOgreDataLoader as yol
    import Mesh.ysMeshUtil as ysu

    def test_layout_MeshInfoWnd():
        sub = ybu.Subject()
        w = MeshInfoWnd(400,400,200,500,sub)
        w.size_range(100,100)
        w.show()
        Fl.run()

    def test_ObjectInfoWnd():

        bvhFilePath = '../samples/wd2_WalkSameSame00.bvh'
        jointMotion1, frameTime = yf.readBvhFileAsJointMotion(bvhFilePath, .01)
        bvhFilePath = '../samples/wd2_WalkForwardVFast00.bvh'
        jointMotion2, frameTime = yf.readBvhFileAsJointMotion(bvhFilePath, .01)

        meshFilePath = '../samples/physics2_woody_binding1.mesh.xml'
        skinMesh, js = yol.readOgreDataFiles(meshFilePath, .01)
        meshFilePath = '../samples/physics2_woody_binding1.mesh.xml'
        mesh = yol.readOgreMeshFileAsMesh(meshFilePath, .01)

        ysu.mergePoints(skinMesh)
        ysu.mergePoints(mesh)

        skinMesh.update(js[0][80])

        viewer = SimpleViewer()
        viewer.doc.addRenderer('motion1(%s)'%jointMotion1.resourceName, yr.JointMotionRenderer(jointMotion1, (0, 0, 255), yr.LINK_LINE))
        viewer.doc.addObject('motion1(%s)'%jointMotion1.resourceName, jointMotion1)
        viewer.doc.addRenderer('motion2(%s)'%jointMotion2.resourceName, yr.JointMotionRenderer(jointMotion2, (0, 0, 255), yr.LINK_LINE))
        viewer.doc.addObject('motion2(%s)'%jointMotion1.resourceName, jointMotion2)

        viewer.doc.addRenderer('skinMesh', yr.MeshRenderer(skinMesh))
        viewer.doc.addObject('skinMesh', skinMesh)
        viewer.doc.addRenderer('mesh', yr.MeshRenderer(mesh))
        viewer.doc.addObject('mesh', mesh)

        viewer.startTimer(frameTime)
        viewer.show()
        Fl.run()

    def test_SimpleViewer():
        #        mmFilePath = '../samples/physics2_WalkSameSame01.mm'
        #        pointMotion1 = yf.readMMFile(mmFilePath)
        #        mmFilePath = '../samples/physics2_WalkForwardFast00.mm'
        #        pointMotion2 = yf.readMMFile(mmFilePath)
        #        frameTime = 1./30.
        #
        #        viewer = SimpleViewer()
        ##        viewer = SimpleViewer((200,200,400,400))
        #        viewer.doc.addMotion(pointMotion1)
        #        viewer.doc.addMotion(pointMotion2)
        #        viewer.doc.addRenderer('WalkSameSame', yr.PointMotionRenderer(pointMotion1))
        #        viewer.doc.addRenderer('WalkForwardFast', yr.PointMotionRenderer(pointMotion2))
        ##        viewer.renderersWnd.when(FL_WHEN_CHANGED)
        ##        print FL_WHEN_RELEASE
        ##        print viewer.renderersWnd.when()
        #
        #        viewer.startTimer(frameTime)
        #        viewer.show()
        #        Fl.run()
        pointMotion = yf.readTrcFile('../samples/Day7_Session2_Take01_-_walk.trc', .01)
        jointMotion = yf.readBvhFile('../samples/wd2_WalkSameSame00.bvh', .01)

        print('pointSkeleton')
        print(pointMotion[0].skeleton)
        print('jointSkeleton')
        print(jointMotion[0].skeleton)

        viewer = SimpleViewer()
        viewer.record(False)
        viewer.doc.addRenderer('pointMotion', yr.PointMotionRenderer(pointMotion, (0,255,0)))
        viewer.doc.addObject('pointMotion', pointMotion)
        viewer.doc.addRenderer('jointMotion', yr.JointMotionRenderer(jointMotion, (0,255,0)))
        viewer.doc.addObject('jointMotion', jointMotion)

        viewer.startTimer(1/pointMotion.fps)
        viewer.show()

        Fl.run()


    def test_Fl_Check_Browser():
        def cb_callback(ptr):
            print('test')

        win = Fl_Window(100,100,200,200)
        win.begin()
        cb = Fl_Check_Browser(100,100,100,100)
        cb.callback(cb_callback)
        cb.add('test1', 1)
        cb.add('test2', 0)
        win.end()

        win.show()
        Fl.run()

    def test_Fl_Hold_Browser():
        def cb_callback(ptr):
            print('test')

        win = Fl_Window(100,100,200,200)
        win.begin()
        cb = Fl_Hold_Browser(100,100,100,100)
        cb.callback(cb_callback)
        cb.add('test1', 1)
        cb.add('test2', 0)
        win.end()

        win.show()
        Fl.run()

    def test_time_check():
        viewer = SimpleViewer()
        viewer.record(False)
        viewer.setMaxFrame(100)

        pt = [0.]
        def simulateCallback(frame):
            if frame==1: pt[0] = time.time()
            if frame==31: print('elapsed time for 30 frames:', time.time()-pt[0])
        #            time.sleep(0.03)
        viewer.setSimulateCallback(simulateCallback)

        viewer.startTimer((1/30.)*(1/1.4))
        viewer.show()

        Fl.run()

    #    test_layout_MeshInfoWnd()
    #    test_ObjectInfoWnd()
    # test_SimpleViewer()
    # test_Fl_Check_Browser()
    # test_Fl_Hold_Browser()
    test_time_check()
