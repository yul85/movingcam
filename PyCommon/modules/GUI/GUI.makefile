UNAME := $(shell uname -s)
PYVER := $(shell python -V 2>&1 | sed "s/Python \([0-9]*\).\([0-9]*\).\([0-9]*\)/\1.\2/")
#ifeq ($(UNAME), Darwin)
#	MACVER := $(shell sw_vers -productVersion | sed "s:.[[:digit:]]*.$$::g")
#	FOLDER := lib.macosx-$(MACVER)-x86_64-$(PYVER)
#	MAC_OMP := $(shell clang-omp++ --version 2>/dev/null)
#ifdef MAC_OMP
#	SETUPFILE := setup_mac_omp.py
#	SETUPARG := --with-mac-omp
#else
#	SETUPFILE := setup_mac.py
#	SETUPARG := --with-mac
#endif
#endif
#ifeq ($(UNAME), Linux)
#	FOLDER := lib.linux-x86_64-$(PYVER)
#	SETUPFILE := setup.py
#endif

ifeq ($(UNAME), Darwin)
	MACVER := $(shell sw_vers -productVersion | sed "s:.[[:digit:]]*.$$::g")
	FOLDER := lib.macosx-$(MACVER)-x86_64-$(PYVER)
	SETUPARG := --with-mac-omp
else ifeq ($(UNAME), Linux)
	FOLDER := lib.linux-x86_64-$(PYVER)
endif
SETUPFILE := setup.py

all:
	#python setup.py build ; cp build/lib.linux-x86_64-2.7/*.so ./
	python $(SETUPFILE) build $(SETUPARG); cp build/$(FOLDER)/*.so ./

clean:
	rm -rf build

