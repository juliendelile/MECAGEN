## MECAGEN - A Simulation Platform of Animal Embryogenesis

[MECAGEN](http://www.mecagen.org) aims to investigate the multiscale dynamics of the early stages of biological morphogenesis. 

### Getting Started

#### Install on Linux

##### Dependencies

Installing MECAGEN requires a few libraries to be installed first. The provided APT command lines are operative on Ubuntu/Debian but alternative installation method can be used (tested with pacman on Archlinux).

* SDL2 (not to be mistaken with sdl1.2)
```
sudo apt-get install libsdl2-dev
```

* GLEW

    sudo apt-get install libglew-dev

* GLUT

    sudo apt-get install freeglut3-dev

* Boost Serialization 

    sudo apt-get install libboost-serialization-dev

* Boost Random Number Library

    sudo apt-get install libboost-random-dev

* QT5

From the [QT project webpage](http://qt-project.org/downloads) or directly

	sudo apt-get install qt5-default

The QT library folder must be in the LIBRARY_PATH environment variable and the folder containing the "moc" program in the PATH environment variable.

* CUDA 5.5 or newer

Cuda is required to enable an enhanced rendering of the simulations. It uses Vertex Buffer Object to interoperate with OpenGL. 

    sudo apt-get install nvidia-cuda-toolkit

* [Thrust](http://thrust.github.io/)

Thrust is included with Cuda. Yet if you install MECAGEN without Cuda, the Thrust header library must be installed manually. Thrust is a C++ template library so the files just need to be copied somewhere on your system.



##### Compilation

modify makefile 
cuda path 
thrust path

version with VBO rendering

version with VBO rendering. Better graphics. 

%  make USE_VBO=1

version without VBO rendering.

%  make USE_VBO=1

#### Compile and run examples

All examples are compiled by:


generate xml files

écrire make example script


#### Write new simulations

auie

### License

MECAGEN is released under the GNU General Public License v3.0; see LICENSE for more details.

