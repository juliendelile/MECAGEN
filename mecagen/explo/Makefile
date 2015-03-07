# mar. janv. 14 18:01:02 CET 2014
# MecaGen Team

# Compiler:
CC      = g++
CFLAGS  += -fpic

# Files:
INC = ./includes
SRC = ./src

########################################################################################################################
# Headers dependences:
HEADERS = $(wildcard $(INC)/*.hpp)
# Object to build:
OBJECTS = $(patsubst %.cpp, %.o, $(wildcard $(SRC)/*.cpp))
########################################################################################################################

.PHONY: all clean propper

all: bin/mecagenEXPLO

bin/mecagenEXPLO: $(OBJECTS)
	$(CC) $(OBJECTS) -L$(MODELPATH)/lib/$(THRUSTMODE) -lmodel -L$(CONSUMERSPATH)/lib/$(THRUSTMODE) -lconsumers -L$(PRODUCERSPATH)/lib/$(THRUSTMODE) -lproducers $(LIB) -o $@

$(OBJECTS): %.o: %.cpp $(HEADERS)
	$(CC) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP $(CFLAGS) -I$(INC) -c $< -o $@

########################################################################################################################

clean:
	rm -f $(OBJECTS)

propper: clean
	rm -f bin/mecagenEXPLO




