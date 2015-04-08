


# **************************************
# EDIT THE FOLLOWING SECTION: QT/Cuda/Thrust paths
# **************************************

# QT (Edit required):
# Write yout QT "include" and "lib" paths here:
export QTINCLUDEPATH ?= /usr/include/qt5
export QTLIBPATH ?= /usr/lib/x86_64-linux-gnu/qt5

# Edit on of the following options, and comment the other

# Option 1: 
# Write your cuda path here if you plan to use the Cuda/VBO rendering
export CUDAPATH ?= /usr/local/cuda
export THRUSTPATH  ?= $(CUDAPATH)/include

# Option 2:
# Write below the path which contains the thurst folder if you do not plan to use the Cuda/VBO rendering.
# export THRUSTPATH ?= /data/Work/Applications/libraries

# **************************************
# END OF SECTION REQUIRING EDITING
# **************************************



########################################################################################################################
# Project version:
########################################################################################################################
# Mecagen/shared object version.
export MCG_VERSION  := 1
export MCG_MINOR    := 0
export MCG_RELEASE  := 0

########################################################################################################################
# Project paths:
########################################################################################################################
# Common:
export COMPATH        := $(abspath ./common)
# Components path:
export MODELPATH      := $(abspath ./model)
export PRODUCERSPATH  := $(abspath ./producers)
export CONSUMERSPATH  := $(abspath ./consumers)

########################################################################################################################
# Project compilation & linking variables:
########################################################################################################################
# CFLAGS contains compilation flags: compiler options and headers directories (includes paths).
# export CFLAGS := -g
export CFLAGS := -O3
# export CFLAGS := 
CFLAGS += -I$(COMPATH)/includes
CFLAGS += -I$(MODELPATH)/includes
CFLAGS += -I$(PRODUCERSPATH)/includes
CFLAGS += -I$(CONSUMERSPATH)/includes

# Linking flags:
export LIB
export LIBCUDA += -L$(CUDAPATH)/lib64

########################################################################################################################
# Cuda Compiler:
########################################################################################################################
# WARNING: add -DBOOST_NOINLINE='__attribute__ ((noinline))' so the compilation succeed while using boost serialiation
# export CUDACC    ?= nvcc -gencode=arch=compute_11,code=\"sm_11,compute_11\" -DBOOST_NOINLINE='__attribute__ ((noinline))'
# export CUDACC    ?= nvcc -gencode=arch=compute_20,code=\"sm_20,compute_20\" -DBOOST_NOINLINE='__attribute__ ((noinline))'
export CUDACC    ?= nvcc -gencode=arch=compute_30,code=\"sm_30,compute_30\" -DBOOST_NOINLINE='__attribute__ ((noinline))'

########################################################################################################################
# Project third-party configuration:
########################################################################################################################
#### ISF:
# Path to the ISF directory
export ISFPATH  := $(abspath ../isf)
export ISFLIB   := $(ISFPATH)/lib/libisf.a   # Static linking of ISF
CFLAGS          += -I$(ISFPATH)/includes     # Add ISF include path in CFLAGS
LIB             += $(ISFLIB)                 # Add ISF lib

#### SDL:
CFLAGS  += $(shell sdl2-config --cflags) # SDL includes & compilation flags
LIB     += $(shell sdl2-config --libs)   # SDL linking flags

#### Thrust:
CFLAGS  += -I$(THRUSTPATH)  # Add Thrust includes in CFLAGS

## gcc/nvcc -D preprocessor option for THRUST backend:
# This allow to configure Thrust
export THRUSTD
THRUSTSYSTEMCPP  := THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
THRUSTSYSTEMOMP  := THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
THRUSTSYSTEMCUDA := THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

## Thrust mode (cpp, cuda or omp):
# This is related to the above and is also used to create directory like ./lib/$(THRUSTMODE)
# i.e. ./lib/cpp, ./lib/omp and ./lib/cuda (cf the OUTPATH variable)
export THRUSTMODE
export THRUSTCPP  := cpp
export THRUSTOMP  := omp
export THRUSTCUDA := cuda

#### Boost:
export BOOSTLIB += -lboost_serialization
LIB             += $(BOOSTLIB)

########################################################################################################################
# Configuration for generated object according to cuda/omp/cpp mode
########################################################################################################################
# Here, 'mode' refers to the compilation mode: cpp, cuda or omp.
export OUTPATH      # Contains the output path of some components according to the mode
export DEVICE_CC    # Contains the compiler  according to the mode
export DEVICE_FLAGS # Contains the compiler flags according o the mode
export DEVICE_LIB   # Contains the libraries to link against according to the mode
export DEVICE_SFX   # Source extension to compile according to the mode

########################################################################################################################
# Configuration for the gui (CUDA_VBO or not)
########################################################################################################################
# To compile with the VBO, just define the variable when invoking make, e.g. make cpp USE_VBO=1
ifdef USE_VBO
	CFLAGS += -DCUDA_VBO                    # Activate the related #define in source code (cf ./gui/src/glwidget.cpp)
	export USE_VBO                          # Transmit the info to submake (Note: not use for the moment)
	export VBOLIB := $(LIBCUDA) -lcuda -lcudart    # Define the libraries to link againt when building with VBO, i.e. cuda
endif

########################################################################################################################
# Configuration for the customized sections of the code (default, ...)
########################################################################################################################

# Initialize CUSTOM as "default" if it is undefined
export CUSTOM             ?= default
export CUSTOMPATH_ROOT    =  $(abspath ./custom)/$(CUSTOM)
export CUSTOMPATH_MODEL   =  $(CUSTOMPATH_ROOT)/model
export CUSTOMPATH_GUI     =  $(CUSTOMPATH_ROOT)/gui
export CUSTOM_DEP_MODEL   += $(shell find $(CUSTOMPATH_MODEL) -type f)
export CUSTOM_DEP_GUI     += $(shell find $(CUSTOMPATH_GUI) -type f)
export CUSTOM_DIRS        =  $(shell find $(abspath ./custom) -mindepth 1 -maxdepth 1 -type d)
CFLAGS += -I$(CUSTOMPATH_MODEL)/includes
CFLAGS += -I$(CUSTOMPATH_GUI)/includes
LIB    += $(CUSTOMPATH_MODEL)/libcustom.a

########################################################################################################################
# Rules:
########################################################################################################################
# Phony rules do not depend on files or directories (i.e. do not look in the filesystem for dependencies)
.PHONY: cuda omp cpp build gui explo clean propper

# By default, build everything in cpp mode
all: cpp

# Here are our 3 mode: cuda, omp and cpp.
# Each mode define accordingly several variables used by other part of the build process.
# Note: it builds both the exploration version and the gui version of MecaGen

# Set the cuda mode and put the correct -D flags for thrust in THRUSTD
cuda: THRUSTMODE    = $(THRUSTCUDA)
cuda: THRUSTD       = $(THRUSTSYSTEMCUDA)
# We specify the compiler, its flags and the needed libraries:
cuda: DEVICE_CC     = $(CUDACC)
cuda: DEVICE_FLAGS  = --compiler-options '-fpic' -I$(CUDAPATH)/include
cuda: DEVICE_LIB    = -L$(CUDAPATH)/lib64/ -lcudart -lcuda -ldl
cuda: DEVICE_SFX    = cu
# Call the next step of the build process
cuda: build

omp: THRUSTMODE     = $(THRUSTOMP)
omp: THRUSTD        = $(THRUSTSYSTEMOMP)
omp: DEVICE_CC      = g++
omp: DEVICE_FLAGS   = -fpic -fopenmp
omp: DEVICE_LIB     = -lgomp
omp: DEVICE_SFX     = cpp
omp: LIB 						+= -fopenmp 
omp: build

cpp: THRUSTMODE     = $(THRUSTCPP)
cpp: THRUSTD        = $(THRUSTSYSTEMCPP)
cpp: DEVICE_CC      = g++
cpp: DEVICE_FLAGS   = -fpic
cpp: DEVICE_LIB     =
cpp: DEVICE_SFX     = cpp
cpp: build

# Set the output path for shared objects (components) and build the exploration version and the gui version
build: OUTPATH    = ./lib/$(THRUSTMODE)
build: gui
#build: gui explo

# The gui version depends on components 
gui: components gui/Makefile
	$(MAKE) -C ./gui

# The exploratio version only depends on components
explo: components
	$(MAKE) -C ./explo

########################################################################################################################
# Compiler flags
########################################################################################################################
# The gui version is the one depending on the USE_VBO variable. We want to recompile our gui version if the variable
# changes between to "make" invocation. To achieve this, we save the compilation flags in the "compiler_flags" file.
# On subsequent invocations, we compare our current flags string we the one we stored
#  --- this is the    echo '$(CFLAGS)' | cmp -s - $@    part below, 'man cmp' for more info on the cmp command;
# if the flags are identical, the following part is not called as the || is a lazy or operator
# (i.e. true || whatever is true, hence do not call "whatever").
# However, if flags differ then we need to recompile our gui. As Qmake is a bit painfull with this, we simplify the
# process by "touching" the glwidget source file, acting as if it was modified and hence triggering a recompilation.
# We also update the compiler_flags with the new flags.
#
# The same goes for other compiler flags

# .PHONY tells make to ignore the compiler_flags file when checking dependencies, hence actually executing this
# target on each invocation regardless of the file.
.PHONY: force
compiler_flags_vbo: force
	echo '$(USE_VBO)' | cmp -s - $@ || `echo '$(USE_VBO)' > $@;  touch ./gui/src/glwidget.cpp`

# compiler_flags_mode: force
# 	echo '$(MODE)' | cmp -s - $@ || `echo '$(MODE)' > $@;  touch ./gui/src/*`

########################################################################################################################
# COMPONENTS part:
########################################################################################################################
# Components are the producers, the consumers and the model itself
# components: compiler_flags_mode
components: 
	$(MAKE) all -C $(CUSTOMPATH_MODEL)
	$(MAKE) all -C $(MODELPATH)
	$(MAKE) all -C $(PRODUCERSPATH)
	$(MAKE) all -C $(CONSUMERSPATH)

########################################################################################################################
# GUI MAKEFILE
########################################################################################################################
# The Gui makefile is 100% cpp: Check ./gui/QMakefile.pro !
# Go in the ./gui dit and execute qmake with parameters:
# Add our CFLAGS to the QMAKE flags
# Specify our components libraries
# Specify the ISF library
# Specify supplementary libraries needed by the VBO version
gui/Makefile: components compiler_flags_vbo
	$(MAKE) all -C ./gui

########################################################################################################################
# Cleaning:
########################################################################################################################
# "clean" removes "*.o" files and "propper" removes everything that is generated

# Just execute the "clean" target of submakes and remove the compiler_flags file
# The dash "-" in front of command tells make to ignore errors (e.g. continue even if a rm command failed)
clean:
	- rm ./compiler_flags_vbo
	# - rm ./compiler_flags_mode
	- $(MAKE) clean -C ./gui
	- $(MAKE) clean -C ./explo
	- $(MAKE) clean -C $(MODELPATH)
	- $(MAKE) clean -C $(PRODUCERSPATH)
	- $(MAKE) clean -C $(CONSUMERSPATH)
	- for i in $(CUSTOM_DIRS); do $(MAKE) clean -C $$i/model; done
	- for i in $(CUSTOM_DIRS); do $(MAKE) clean -C $$i/gui; done

# Same thing for propper
propper: clean
	- $(MAKE) propper   -C ./gui        # Qt distclean command
	- $(MAKE) propper   -C ./explo
	- $(MAKE) propper   -C $(MODELPATH)
	- $(MAKE) propper   -C $(PRODUCERSPATH)
	- $(MAKE) propper   -C $(CONSUMERSPATH)
	- for i in $(CUSTOM_DIRS); do $(MAKE) propper -C $$i/model; done
	- for i in $(CUSTOM_DIRS); do $(MAKE) propper -C $$i/gui; done
	- $(MAKE) propper -C ./examples/default/Boundaryformation_epithelialization/param
	- $(MAKE) propper -C ./examples/default/Boundaryformation_epithelialization/state
	- $(MAKE) propper -C ./examples/default/BoxTissueFrom1cell_grn_default/state
	- $(MAKE) propper -C ./examples/default/BoxTissueFrom1cell_grn_default/param
	- $(MAKE) propper -C ./examples/default/SignalMedToggleSwitch_FlatTissue_grn_default/state
	- $(MAKE) propper -C ./examples/default/SignalMedToggleSwitch_FlatTissue_grn_default/param
	- $(MAKE) propper -C ./examples/default/FlatTissueFrom1cell_grn_default/param
	- $(MAKE) propper -C ./examples/default/FlatTissueFrom1cell_grn_default/state
	- $(MAKE) propper -C ./examples/zebrafish/Zebrafish_epiboly
