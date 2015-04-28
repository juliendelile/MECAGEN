## MecaGen - A Simulation Platform of Animal Embryogenesis

[MecaGen](http://www.mecagen.org) is a C++ simulation platform of animal multicellular development relying on a realistic agent-based model. It is centered on the physico-chemical coupling of cell mechanics with gene expression and molecular signaling.

This project aims to investigate the multiscale dynamics of the early stages of biological morphogenesis. Embryonic development is viewed as an emergent, self-organized phenomenon based on a myriad of cells and their genetically regulated, and regulating, biomechanical behavior.

### 1. Installing on Linux

#### 1.1. Dependencies

(a) Installing MecaGen requires a few libraries to be installed first. We recommend using a package manager such as "pacman" on Archlinux, or the "Advanced Packaging Tool" (APT) on Ubuntu and Debian. For the last two, you can copy and paste the following command in a terminal window:

```shell
sudo apt-get install libsdl2-dev libglew-dev freeglut3-dev libboost-serialization-dev libboost-random-dev qt5-default
```

These libraries can also be installed manually via the download links below:

* <a href="https://www.libsdl.org/release/SDL2-2.0.3.tar.gz" target="_blank">Simple DirectMedia Layer (SDL 2.0)</a>
* <a href="https://sourceforge.net/projects/glew/files/glew/1.12.0/glew-1.12.0.tgz/download" target="_blank">The OpenGL Extension Wrangler Library (GLEW)</a>
* <a href="http://sourceforge.net/projects/freeglut/files/freeglut/3.0.0/freeglut-3.0.0.tar.gz" target="_blank">The OpenGL Utility Toolkit (GLUT)</a>
* <a href="http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz/download" target="_blank">Boost</a>
* <a href="http://download.qt.io/official_releases/online_installers/qt-opensource-linux-x64-online.run" target="_blank">QT5.x</a>

(b) Then, if your machine is equipped with an NVIDIA graphics card, we recommend to install the CUDA library, by command or download:

```shell
sudo apt-get install nvidia-cuda-toolkit
```

* <a href="https://developer.nvidia.com/cuda-toolkit-55-archive" target="_blank">CUDA 5.5 or newer</a>

(c) If you do not have an NVIDIA card, or are not sure, then you must download instead the Thrust template library (a package of header files), and unzip it in the directory of your choice:

* (Required only if no Cuda is installed) <a href="https://github.com/thrust/thrust/releases/download/1.8.1/thrust-1.8.1.zip" target="_blank">Thrust</a>

#### 1.2. Environment variables

(a) If you installed the above QT5.x library by download, verify that the following environment variables of your system contain the proper library folders:

* LIBRARY_PATH must contain the QT library folder <b><i>(MATTHIEU : A CONFIRMER)</i></b>
* PATH must contain the Meta-Object Compiler (moc) program

(b) If you installed CUDA by download, verify that:

* PATH contains the NVIDIA CUDA Compiler (nvcc)

(c) Edit the custom file containing the user paths "user_paths_MUST_BE_EDITED_FIRST" on your disk as as indicated inside the file.

#### 1.3. Compilation

At this stage, compilation options will depend on the examples you want to run. This is because the MecaGen platform was designed to allow integration with external custom code, in order to simulate specific structures such as extraembryonic tissue. Currently, two versions are available: (a) a "default" version executing regular MecaGen simulations without custom code, and (b) a "zebrafish" version that includes special rules for the yolk particles and enveloping layer (EVL) cells.

(a) Go to the root directory of the unzipped MecaGen package (replace "/path/to/" with the actual path containing the MECAGEN-master folder):

```shell
cd /path/to/MECAGEN-master
```

(b) To compile MecaGen in the "default" mode and run Case Study 1 (pattern formation) or Case Study 2 (epithelial differentiation), enter the following command:

```shell
make CUSTOM=default
```

(c) To compile MecaGen in the "zebrafish" mode, and run Case Study 3 (epiboly), enter:

```shell
make CUSTOM=zebrafish
```

(d) If you need to switch modes to run another case study, you must clean all previous compilation files beforehand by typing:

```shell
make cleanall
```

### 2. Running examples

MecaGen uses XML files as inputs for the simulations. It requires three input files, which can be automatically generated (see below):
- a *parameter* file containing all the parameters describing the gene regulatory network (GRN), molecular interactions, and biomechanical properties
- a *state* file containing the initial state of the variables
- a *meta-parameter* file containing information related to algorithmic and rendering options

#### 2.1. Generating input files

For simulations containing a large number of cells, it is not practical to write the input files by hand. Instead, you can generate them automatically by typing the following commands:

```shell
cd mecagen
./generate_input_files.sh all
```

#### 2.2. Run examples

Once input files are generated, MecaGen simulations can be started by running the [run_examples.sh](mecagen/run_examples.sh) script. If the execution returns prematurely with "gpuassert", please start the script with sudo: "sudo run_examples.sh".

#### 2.3. Support videos

Two videos demonstrate the previous instructions and are accessible on youtube:

* the [first video](https://www.youtube.com/watch?v=d79v7MDPIBw) details the procedure required to compile and run the "zebrafish" mode simulation with Cuda/VBO rendering.
* the [second video](https://www.youtube.com/watch?v=5zcLAL-caDQ) details the procedure required to compile and run the "default" mode simulations without the Cuda/VBO rendering. 

## 3. License

MecaGen is released under the GNU General Public License v3.0; see LICENSE for more details.