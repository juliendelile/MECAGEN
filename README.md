# MECAGEN - A Simulation Platform of Animal Embryogenesis

[MECAGEN](http://www.mecagen.org) aims to investigate the multiscale dynamics of the early stages of biological morphogenesis. 

## Getting Started

### Install

#### Dependencies

Installing MECAGEN requires a few libraries to installed first. 

version with VBO rendering. Better graphics. 

#### Compilation

modify makefile 
cuda path 
thrust path

%  make USE_VBO=1

version without VBO rendering.

%  make USE_VBO=1

### Compile and run examples

All examples are compiled by:


generate xml files

écrire make example script


### Write new simulations

auie

## License

MECAGEN is released under the GNU General Public License v3.0; see LICENSE for more details.




** Thurst 

if you have cuda > XX ok 

if not copy header somewher


** missing headers

sdl.h

-> sudo apt-get install libsdl2-dev (pas libsdl1.2)

GL/glew.h

-> sudo apt-get install libglew-dev

GL/glut.h

-> sudo apt-get install freeglut3-dev

 boost/random/uniform_real_distribution.hpp

sudo apt-get install libboost-random-dev


** missing lib

-lboost_serialization

-> sudo apt-get install libboost-serialization-dev

*** CUDA ***

install at least CUDA 5.5

directly from .run file or sudo apt-get install nvidia-cuda-toolkit

**** QT ****

Qt Online Installer for Linux 64-bit (22 MB) from http://qt-project.org/downloads

or sudo apt-get install qt5-default (automatically deals with QT environment variable settings)


* if errors during compilation:

/usr/bin/ld: cannot find -lQt5OpenGL
/usr/bin/ld: cannot find -lQt5Widgets
/usr/bin/ld: cannot find -lQt5Gui
/usr/bin/ld: cannot find -lQt5Core

-> check that the Qt lib folder is in the LIBRARY_PATH environment variable

* if error during compilation: 

"could not exec '/..../..../moc': No such file or directory"

-> check that the folder containing "moc" program is in the PATH environment variable-
* if error at runtime:

error while loading shared libraries: libQt5OpenGL.so.5: cannot open shared object file: No such file or directory

-> 
