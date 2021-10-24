# Human Dynamics from Monocular Video with Dynamic Camera Movements

[Ri Yu](https://yul85.github.io), [Hwangpil Park](https://hpgit.github.io) and [Jehee Lee](https://mrl.snu.ac.kr/~jehee)

Seoul National University

ACM Transactions on Graphics, Volume 40, Number 6, Article 208. (SIGGRAPH Asia 2021)

![Teaser Image](figures/teaser_parkour14.png)

## Abstract
We propose a new method that reconstructs 3D human motion from in-the wild video by making full use of prior knowledge on the laws of physics. Previous studies focus on reconstructing joint angles and positions in the body local coordinate frame. Body translations and rotations in the global reference frame are partially reconstructed only when the video has a static camera view. We are interested in overcoming this static view limitation to deal with dynamic view videos. The camera may pan, tilt, and zoom to track the moving subject. Since we do not assume any limitations on camera movements, body translations and rotations from the video do not correspond to absolute positions in the reference frame. The key technical challenge is inferring body translations and rotations from a sequence of 3D full-body poses, assuming the absence of root motion. This inference is possible because human motion obeys the law of physics. Our reconstruction algorithm produces a control policy that simulates 3D human motion imitating the one in the video. Our algorithm is particularly useful for reconstructing highly dynamic movements, such as sports, dance, gymnastics, and parkour actions.


## Requirements 

* Ubuntu (tested on 18.04 LTS)

* Python 3 (tested on version 3.6+)

* Dart (modified version, see below)

* Fltk 1.3.4.1

## Installation

**Dart**

    sudo apt install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev libopenscenegraph-dev libnlopt-dev coinor-libipopt-dev libbullet-dev libode-dev liboctomap-dev libflann-dev libtinyxml2-dev liburdfdom-dev doxygen libxi-dev libxmu-dev liblz4-dev
    git clone https://github.com/hpgit/dart-ltspd.git
    cd dart-ltspd
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
    sudo apt install libgle3-dev


## Run examples

    source venv/bin/activate
    export PYTHONPATH=$PWD
    cd control/parkour1
    python3 render_parkour1.py


## Bibtex

    @article{Yu:2021:MovingCam,
        author = {Yu, Ri and Park, Hwangpil and Lee, Jehee},
        title = {Human Dynamics from Monocular Video with Dynamic Camera Movements},
        journal = {ACM Trans. Graph.},
        volume = {40},
        number = {6},
        year = {2021},
        articleno = {208}
    }
