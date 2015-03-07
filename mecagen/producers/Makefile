# 2014-01-07
# MecaGen Team

########################################################################################################################
# Library configuration:
########################################################################################################################
# Put the name of the components being built
NAME        = libproducers
# Genererate other names according to the above name:
SONAME      = $(NAME).so
SONAME_V    = $(SONAME).$(MCG_VERSION)
SOFULLNAME  = $(SONAME_V).$(MCG_MINOR).$(MCG_RELEASE)
# Generate our main target depending of the above name and received output path
# (OUTPATH is exported from the main Makefile)
SOTARGET    = $(OUTPATH)/$(SOFULLNAME)
# Create our "so" command:
SO          = gcc -shared -Wl,-soname,$(SONAME_V),--whole-archive

########################################################################################################################
# Header and source files:
########################################################################################################################
INC = ./includes
SRC = ./src

# Headers dependencies:
HEADERS  = $(wildcard $(INC)/*.hpp)               # Our headers
HEADERS += $(wildcard $(MODELPATH)/*.hpp)         # Model headers
HEADERS += $(wildcard $(ISFPATH)/includes/*.hpp)  # ISF headers
HEADERS += $(wildcard $(COMPATH)/includes/*.hpp)  # MecaGen common headers
HEADERS += $(CUSTOM_DEP_MODEL)

########################################################################################################################
# Conditionnal compiler & rules
########################################################################################################################
.PHONY: all clean propper

# CFLAGS extended with include flags from producers and from model
CFLAGS += -I$(INC) -I$(MODELPATH)/includes
# Thrust macro must be first in order to play its role in includes: rewrite the flags with it in front of others.
FLAGS = -D$(THRUSTD) $(DEVICE_FLAGS) $(CFLAGS)

# Objects to build, based on .$(DEVICE_SFX) files, specialized for the mode,
# e.g. with myfile.cpp and myfile.cu being a symlink to myfile.cpp (for nvcc),
# myfile_cpp.o  in cpp mode,
# myfile_omp.o  in omp mode,
# myfile_cuda.o in cuda mode
OBJECTS = $(patsubst %.$(DEVICE_SFX), %_$(THRUSTMODE).o, $(wildcard $(SRC)/*.$(DEVICE_SFX)))

$(OBJECTS): %_$(THRUSTMODE).o: %.$(DEVICE_SFX) $(HEADERS)
	$(DEVICE_CC) $(FLAGS) -c $< -o $@


########################################################################################################################
# General rules:
########################################################################################################################
all: $(SOTARGET)

# Build dynamic library and create symbolic links:
$(SOTARGET): $(OBJECTS) $(OUTPATH)
	$(SO) $(DEVICE_LIB) $(LIB) $(OBJECTS) -Wl,--no-whole-archive -o $@

$(OUTPATH):
	- mkdir -p $(OUTPATH)
	- cd $(OUTPATH); ln -s $(SOFULLNAME) $(SONAME)
	- cd $(OUTPATH); ln -s $(SOFULLNAME) $(SONAME_V)

########################################################################################################################
# Cleaning:
########################################################################################################################
clean:
	rm -f $(SRC)/*.o

propper: clean
	rm -rf ./lib
