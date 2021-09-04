from . import ysSimpleViewer_ori as ysvOri
from ..GUI import ysBaseUI as ybu
import fltk
try:
    # for python3
    import pickle
except:
    # for python2.7
    import cPickle as pickle


class DartViewer(ysvOri.SimpleViewer):
    def __init__(self, rect=None, title='DartViewer'):
        ybu.BaseWnd.__init__(self, rect, title, ysvOri.SimpleSetting())
        self.title = title
        self.doc = ysvOri.SimpleDoc()
        self.begin()
        self.panelWidth = 300
        panelWidth = self.panelWidth
        t = .2
        self.renderersWnd = ysvOri.RenderersWnd(self.w()-panelWidth, 0, panelWidth, int(self.h()*t), self.doc)
        self.objectInfoWnd = DartObjectInfoWnd(self.w() - panelWidth, int(self.h() * t), panelWidth, int(self.h() * (1 - t)), self.doc)
        self.motionViewWnd = None  # type: DartMotionViewWnd
        self.cForceWnd = None
        self.motionViewWnd = DartMotionViewWnd(0, 0, self.w() - panelWidth, self.h(), self.doc)
        self.end()
        self.resizable(self.motionViewWnd)
        self.size_range(600, 400)

        self.motionViewWnd.cForceWnd = None
        self.objectInfoWnd.viewer = self


class DartMotionViewWnd(ysvOri.MotionViewWnd):
    def __init__(self, x, y, w, h, doc):
        ysvOri.MotionViewWnd.__init__(self, x, y, w, h, doc)
        self.mov = False

    def goToFrame(self, frame):
        super(DartMotionViewWnd, self).goToFrame(frame)
        if self.cForceWnd is not None:
            self.cForceWnd.redraw()


class DartObjectInfoWnd(ysvOri.ObjectInfoWnd):
    def __init__(self, x, y, w, h, doc):
        super(DartObjectInfoWnd, self).__init__(x, y, w, h, doc)
        self.valObjects = dict()
        self.valObjOffset = 30

        self.begin()
        saveBtn = fltk.Fl_Button(10, self.valObjOffset, 80, 20, 'param save')
        saveBtn.callback(self.save)
        loadBtn = fltk.Fl_Button(100, self.valObjOffset, 80, 20, 'param load')
        loadBtn.callback(self.load)
        self.end()
        self.valObjOffset += 40

    def update(self, ev, doc):
        super(DartObjectInfoWnd, self).update(ev, doc)

    def addValObjects(self, obj):
        self.valObjects[obj.name] = obj
        pass

    def getValobject(self, name):
        return self.valObjects[name]
        pass

    def getValObjects(self):
        return self.valObjects.values()

    def getVals(self):
        return (v.value() for v in self.valObjects.values())

    def getVal(self, name):
        try:
            return self.valObjects[name].value()
        except Exception as e:
            print(e)
            return 0

    def getNameAndVals(self):
        objValDict = dict()
        for k, v in self.valObjects.items():
            objValDict[k] = v.value()
        return objValDict

    def setVal(self, name, val):
        try:
            self.valObjects[name].value(val)
        except Exception as e:
            print(e)

    def addBtn(self, name, callback):
        self.begin()
        btn = fltk.Fl_Button(10, self.valObjOffset, 80, 20, name)
        btn.callback(callback)
        self.end()
        self.valObjOffset += 40

    def add1DSlider(self, name, minVal, maxVal, valStep, initVal):
        self.begin()
        slider = fltk.Fl_Hor_Value_Slider(10, self.valObjOffset, self.viewer.panelWidth - 30, 18, name)
        slider.textsize(8)
        slider.bounds(minVal, maxVal)
        slider.value(initVal)
        slider.step(valStep)
        slider.label(name)
        slider.name = name
        self.end()
        self.addValObjects(slider)
        self.valObjOffset += 40

    def add1DRoller(self, name):
        class hpRoller(fltk.Fl_Roller):
            def handle(self, event):
                if self.handler is not None:
                    self.handler(self, event)
                return super(hpRoller, self).handle(event)
            def set_handler(self, handler):
                self.handler = handler


        self.begin()
        roller = hpRoller(10, self.valObjOffset, self.viewer.panelWidth - 30, 18, name)
        roller.type(fltk.FL_HORIZONTAL)
        roller.bounds(-1., 1.)
        roller.value(0.)
        roller.step(0.001)
        roller.label(name)
        roller.handler = None
        roller.name = name
        self.end()
        self.addValObjects(roller)
        self.valObjOffset += 40

    def add3DSlider(self, name, minVal, maxVal, valStep, initVal):
        self.begin()

        self.end()
        pass

    def save(self, obj):
        f = open(self.viewer.title+'.param', 'wb')
        pickle.dump(self.getNameAndVals(), f)
        f.close()

    def load(self, obj):
        filefile = fltk.Fl_File_Chooser('.', '*.param', fltk.FL_SINGLE, 'load parameter file')
        filefile.show()
        while filefile.shown():
            fltk.Fl.wait()
        if filefile.count() == 1:
            # f = file(self.viewer.title+'param', 'r')
            f = open(filefile.value(), 'rb')
            objVals = pickle.load(f)
            f.close()
            for k, v in objVals.iteritems():
                if k in self.valObjects.keys():
                    self.valObjects[k].value(v)
