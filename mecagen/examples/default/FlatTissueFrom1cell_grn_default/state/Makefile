
export CUDAPATH ?= /usr/local/cuda
export THRUSTPATH  ?= $(CUDAPATH)/include

export CUSTOM             =  default
export CUSTOMPATH_ROOT    =  $(abspath ../../../../custom)/$(CUSTOM)
export CUSTOMPATH_MODEL   =  $(CUSTOMPATH_ROOT)/model

all: main.o
	g++ -std=c++11 main.o -o genState -lboost_serialization

main.o: main.cpp
	g++ -std=c++11 -c main.cpp -o main.o -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DGRN -I$(THRUSTPATH) -I../../../../common/includes -I../../../../model/includes -I$(CUSTOMPATH_MODEL)/includes

run: all
	./genState