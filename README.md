# Human Dynamics from Monocular Video with Dynamic Camera Movements


![Teaser Image](figures/teaser_parkour14.png)


## Requirements 

* Ubuntu (tested on 18.04 LTS)

* Python 3 (tested on version 3.6+)

* Dart (modified version, see below)

* Fltk 1.3.4.1

## Installation

**Dart**

    sudo apt install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev libopenscenegraph-dev libnlopt-dev coinor-libipopt-dev libbullet-dev libode-dev liboctomap-dev libflann-dev libtinyxml2-dev liburdfdom-dev doxygen libxi-dev libxmu-dev liblz4-dev`
    git clone https://github.com/hpgit/dart.git
    cd dart
    mkdir build
    cd build
    cmake ..
    make -j4
    sudo make install
  

**Pydart**

    sudo apt install swig

after virtual environment(venv) activates,

    source venv/bin/activate
    git clone https://github.com/hpgit/pydart2.git
    cd pydart2
    pip install pyopengl==3.1.0 pyopengl-accelerate==3.1.0
    python setup.py build
    python setup.py install


**Fltk and Pyfltk**

    sudo apt install libfltk1.3-dev

Download [pyfltk](https://sourceforge.net/projects/pyfltk/files/pyfltk/pyFltk-1.3.4.1/pyFltk-1.3.4.1_py3.tar.gz/download)

    cd ~/Downloads
    tar xzf pyFltk-1.3.4.1_py3.tar
    cd pyFltk-1.3.4.1_py3
    python setup.py build
    python setup.py install


**misc**

    pip install pillow cvxopt scipy
    cd PyCommon/modules/GUI
    make -f GUI.makefile
    sudo apt install libgle3-dev


## Run examples

    source venv/bin/activate
    export PYTHONPATH=$PWD
    cd control/parkour1
    python3 render_parkour1.py
