# 2014-01-07
# MecaGen Team

########################################################################################################################
# Tools
########################################################################################################################
LINKER = g++ -Wl,-O1,--sort-common,--as-needed,-z,relro

########################################################################################################################
# Header and source files:
########################################################################################################################
INC = ./includes
SRC = ./src

# Gather sources from the $(SRC) directory:
SOURCES   = $(wildcard $(SRC)/*.cpp)
# Gather custom sources:
SOURCES   += $(wildcard $(CUSTOMPATH_GUI)/src/*.cpp)
# Gather sources from "to be moc" hpp files from the $(INC) directory:
SOURCES   += $(patsubst %.hpp, moc_%.cpp, $(notdir $(wildcard $(INC)/*.hpp)))
# Gather sources from "to be moc" hpp files from the custom includes directory:
SOURCES   += $(patsubst %.hpp, moc_%.cpp, $(notdir $(wildcard $(CUSTOMPATH_GUI)/includes/*.hpp)))

# Gather header dependencies
HEADERS =  $(wildcard $(INC)/*.hpp)
HEADERS += $(wildcard $(COMPATH)/includes/*.hpp)
HEADERS += $(wildcard $(MODELPATH)/includes/*.hpp)
HEADERS += $(CUSTOM_DEP_GUI)

########################################################################################################################
# Object
########################################################################################################################
### When using VBO, OBJECTS = OBJECTS_CPP + VBO else OBJECTS = OBJECTS_CPP
OBJECTS       = $(patsubst %.cpp, %.o, $(SOURCES))
OBJECTS_CPP   := $(OBJECTS)

### Check VBO:
ifdef USE_VBO
	OBJECTS += ./cuda/vbo.o
endif

########################################################################################################################
# Augmented CFLAGS
########################################################################################################################
# CFLAGS extended with include flags from producers and from model
CFLAGS += -I. -I$(INC) -I$(MODELPATH)/includes
## Use pipe instead of files, stack hardening.
CFLAGS += -pipe -fstack-protector --param=ssp-buffer-size=4
## Qt definitions:
CFLAGS += -D_REENTRANT -fPIE
CFLAGS += -DQT_NO_DEBUG -DQT_OPENGL_LIB -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB
## Qt includes:
CFLAGS += -I$(QTINCLUDEPATH)/
CFLAGS += -I$(QTINCLUDEPATH)/QtOpenGL
CFLAGS += -I$(QTINCLUDEPATH)/QtWidgets
CFLAGS += -I$(QTINCLUDEPATH)/QtGui
CFLAGS += -I$(QTINCLUDEPATH)/QtCore

# Thrust macro must be first in order to play its role in includes: rewrite the flags with it in front of others.
# FLAGS = -D$(THRUSTD) $(CFLAGS)
FLAGS = -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP $(CFLAGS)

########################################################################################################################
# Libraries
########################################################################################################################

# Project libraries:
COMPONENTSLIB += -L$(MODELPATH)/$(OUTPATH) -lmodel
COMPONENTSLIB += -L$(PRODUCERSPATH)/$(OUTPATH) -lproducers
COMPONENTSLIB += -L$(CONSUMERSPATH)/$(OUTPATH) -lconsumers

# Qt libraries:
QTLIBS = -L$(QTLIBPATH) -lQt5OpenGL -lQt5Widgets -lQt5Gui -lQt5Core -lGL -lpthread
# QTLIBS = -lQt5OpenGL -lQt5Widgets -lQt5Gui -lQt5Core -lGL -lpthread

# Put everything together
LIBS = $(LIB)              # Main definitions from upper makefile
LIBS += $(COMPONENTSLIB)
LIBS += $(QTLIBS)
LIBS += -lGLU -lglut -lGLEW -ldl

### Check VBO:
ifdef USE_VBO
	LIBS += $(VBOLIB)
endif

########################################################################################################################
# Rules:
########################################################################################################################
.PHONY: all clean propper

### The final step is just linking everything together with the libs
all: $(OBJECTS)
	$(LINKER) $(OBJECTS) -o ./bin/mecagenGUI $(LIBS)
	# $(LINKER) -fpic -fopenmp $(OBJECTS) -o ./bin/mecagenGUI $(LIBS)


### Rule for moc: create them in the current directory from hpp files
moc_%.cpp: $(INC)/%.hpp
	moc $< -o $@

moc_%.cpp: $(CUSTOMPATH_GUI)/includes/%.hpp
	moc $< -o $@

### Rule for compiling OBJECTS_CPP
$(OBJECTS_CPP): %.o: %.cpp $(HEADERS)
	g++ $(FLAGS) -c $< -o $@

### Rule for compiling the VBO
./cuda/vbo.o: ./cuda/kernel.cu ./cuda/kernel.cuh ./cuda/vbo.cu ./cuda/vbo.h
	$(MAKE) all -C ./cuda

########################################################################################################################
# Cleaning:
########################################################################################################################

clean:
	rm -f $(OBJECTS_CPP)
	rm -f moc_*.cpp
	$(MAKE) clean -C ./cuda

propper: clean
	rm ./bin/mecagenGUI
	$(MAKE) propper -C ./cuda
