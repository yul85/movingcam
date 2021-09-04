from datetime import datetime
import operator as op

def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
        
    return L

def mrange(minvec, maxvec=None):  
    if maxvec is None:
        maxvec = minvec
        minvec = [0] * len(maxvec)
    vec = list(minvec)
    unitpos = len(vec) - 1
    maxunit = maxvec[unitpos]
    _tuple = tuple
    while 1:
        if vec[unitpos] == maxunit:
            i = unitpos
            while vec[i] == maxvec[i]:
                vec[i] = minvec[i]
                i -= 1
                if i == -1:
                    return            
                vec[i] += 1
        yield _tuple(vec)
        vec[unitpos] += 1
         
def map_dict(func, dic1, dic2):
    dic = {}
    for key in dic1:
        dic[key] = func(dic1[key], dic2[key])
    return dic

def format_seq(seq, singleLine=False):
    s = ''
    if singleLine:
        s += '['
        for i in range(len(seq)):
            s += seq[i]
        s += ']\n'
    else:
        for i in range(len(seq)):
            s += '[%d]'%i + seq[i].__repr__() + '\n'
    return s

def print_seq(seq, singleLine=False):
    if singleLine:
        print('[')
        for i in range(len(seq)):
            print(seq[i])
        print(']')
    else:
        for i in range(len(seq)):
            print('[%d]'%i, seq[i])
            
def print_dict(dic, singleLine=False):
    if singleLine:
        print('{')
        for key, value in dic.items():
            print('%s'%repr(key), ':', value, ',')
        print('}')
    else:
        for key, value in dic.items():
            print('[%s]'%repr(key), value)
            
def updateObject(object, otherObject):
    object.__dict__ = otherObject.__dict__
    
def getLogTimeString():
    return datetime.today().strftime('%y%m%d_%H%M%S')

def getReverseDict(dic):
    rdic = {}
    for key, value in dic.items():
        rdic[value] = key
    return rdic

# input : 
# inList == [10,20,30]
# outList == [None,None,None,None,None,None]
# repeatNums = [3,2,1]
# output : 
# outLIst == [10,10,10,20,20,30]
def repeatListElements(inList, outList, repeatNums):
    index = 0
    for i in range(len(inList)):
        for j in range(repeatNums[i]):
            outList[index] = inList[i]
            index += 1
    
# input:
# inList == [1,2,3,4,5,6]
# outList == [None, None]
# sumNums == [3,3]       
# output:
# outList == [6, 15] 
def sumListElements(inList, outList, sumNums, addOperation=op.iadd, additiveIdentity=0):
    index = 0
    for i in range(len(sumNums)):
        sum = additiveIdentity
        for j in range(sumNums[i]):
            sum = addOperation(sum, inList[index])
            index += 1
        outList[i] = sum

def makeFlatList(totalDOF):
    return [None]*totalDOF
def makeNestedList(dofs):
    ls = [None]*len(dofs)
    for i in range(len(dofs)):
        ls[i] = [None]*dofs[i]
    return ls
def flatten(nestedList, flatList):
    i = 0
    for a in range(len(nestedList)):
        for b in range(len(nestedList[a])):
            flatList[i] = nestedList[a][b]
            i += 1
def nested(flatList, nestedList):
    i = 0
    for a in range(len(nestedList)):
        for b in range(len(nestedList[a])):
            nestedList[a][b] = flatList[i]
            i += 1




if __name__ == "__main__" :
    import time
    import operator as op

    def test_getLogTimeString():
        print(getLogTimeString())

    def test_frange():
        for i in frange(1.2, 5, 1.):
            print(i)

    def test_mrange():
        for i in mrange([2,2,1,2]):
            print(i)

        #So the rather horrid
        for i in range(1):
            for j in range(2):
                for k in range(3):
                    for l in range(4):
                        print(i, j, k, l)
        #reduces to
        for i, j, k, l in mrange([1,2,3,4]):
            print(i, j, k, l)

    class A:
        def __init__(self, name):
            self.name = name
        def __str__(self):
            return self.name

    def test_print_seq():
        ls = []
        for i in range(10):
            ls.append(A('name_%s'%str(i)))
        print(ls)
        print_seq(ls)

    def test_print_dict():
        dic = {}
        for i in range(10):
            dic[str(i)] = A('name_%s'%str(i))
        #            dic[i] = A('name_%s'%str(i))
        print(dic)
        print_dict(dic)

    def test_updateObject():
        a = A('a')
        b = A('b')
        print(a, b)
        updateObject(a, b)
        print(a, b)

    def test_map():
        def AND_list(ls1, ls2):
            if len(ls1)!=len(ls2):
                raise ValueError
            ls = [None]*len(ls1)
            for i in range(len(ls1)):
                ls[i] = ls1[i] and ls2[i]
            return ls

        ls0 = [1,0,1]
        ls1 = [1,1,0]

        print('list')
        print(map(lambda x, y: x and y, ls0, ls1))
        print(map(op.and_, ls0, ls1))
        print(AND_list(ls0, ls1))
        print()

        dic1 = {0:0, 1:1, 2:2}
        dic2 = {1:1, 2:2, 0:0}
        dic3 = {2:2, 0:0, 'a':4, 1:1}
        del dic3['a']

        print('dict')
        print('dic1', dic1)
        print('dic2', dic2)
        print('dic3', dic3)
        print(map(lambda x, y: x + y, dic1.values(), dic2.values()))
        print(map(lambda x, y: x + y, dic1.values(), dic3.values()))
        print(map_dict(lambda x, y: x + y, dic1, dic3))

    def test_enum():
        class Event:
            EV_addRenderer = 0
            EV_setRendererVisible    = 1
            EV_addObject             = 2
            EV_selectObjectElement   = 3
            EV_selectObject          = 4

            text = getReverseDict(locals())

        print(Event.text)

    pass
    #    test_getLogTimeString()
    #    test_map()
    #    test_frange()
    #    test_updateObject()
    #    test_print_dict()
    #    test_print_seq()
    #    test_mrange()
    test_enum()
