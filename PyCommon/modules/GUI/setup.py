from distutils.core import setup, Extension
import sys
py_major_ver = sys.version_info[0]


class setupmodule:
    def __init__(self, name='noName'):
        self.name = name
        self.include_dirs = ['../usr/include/']
        self.extra_compile_args = []
        self.libraries = []
        self.library_dirs = ['../usr/lib']
        self.extra_link_args = []
        self.sources = [name+'.cpp']
        self.depends = [name+'.h']


isMAC = False
isOMP = True
ompLib = 'gomp'
boost_lib = 'boost_python'

if '--with-mac-omp' in sys.argv:
    isMAC = True
    ompLib = 'omp'
    idx = sys.argv.index('--with-mac-omp')
    sys.argv.pop(idx)
elif '--with-mac' in sys.argv:
    isMAC = True
    isOMP = False
    idx = sys.argv.index('--with-mac')
    sys.argv.pop(idx)

if py_major_ver == 3:
    boost_lib = boost_lib + '3'

modules = []

m = setupmodule('csDump')
m.libraries = [boost_lib]
m.sources.append('lodepng.cpp')
m.depends.append('lodepng.h')
modules.append(m)

for m in modules:
    ext_module = Extension(m.name,
            include_dirs=m.include_dirs,
            extra_compile_args=m.extra_compile_args,
            extra_link_args=m.extra_link_args,
            libraries=m.libraries,
            library_dirs=m.library_dirs,
            sources=m.sources,
            depends=m.depends)

    setup(name=m.name, ext_modules=[ext_module])

