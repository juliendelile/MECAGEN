# 2014-01-07
# MecaGen Team

########################################################################################################################
# Library configuration:
########################################################################################################################
TARGET      = libcustom.a
AR          = ar -rvs $(TARGET)

########################################################################################################################
# Header and source files:
########################################################################################################################
INC = ./includes
SRC = ./src

# Headers dependences:
HEADERS =  $(wildcard $(INC)/*.hpp)               # Our headers
HEADERS += $(wildcard $(ISFPATH)/includes/*.hpp)  # ISF headers
HEADERS += $(wildcard $(COMPATH)/includes/*.hpp)  # MecaGen common headers
HEADERS += $(CUSTOM_DEP)

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
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(AR) $^

########################################################################################################################
# Cleaning:
########################################################################################################################
clean:
	rm -f $(SRC)/*.o

propper: clean
	rm -rf $(TARGET)
