
export CUDAPATH ?= /usr/local/cuda
export THRUSTPATH  ?= $(CUDAPATH)/include

export CUSTOM             =  zebrafish
export CUSTOMPATH_ROOT    =  $(abspath ../../../custom)/$(CUSTOM)
export CUSTOMPATH_MODEL   =  $(CUSTOMPATH_ROOT)/model

all: main.o
	g++ -std=c++11 main.o -o genInputFiles -lboost_serialization

main.o: main.cpp
	g++ -std=c++11 -c main.cpp -o main.o -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -I$(THRUSTPATH) -I../../../common/includes -I../../../model/includes -I$(CUSTOMPATH_MODEL)/includes

run: all
	./genInputFiles

clean:
	rm -f main.o

propper: clean
	rm -f genInputFiles
	rm -f param_archive.xml
	rm -f state_archive.xml
	rm -f metaparam_archive.xml