
# export CUDAPATH ?= /usr/local/cuda
# export THRUSTPATH  ?= $(CUDAPATH)/include

# Use vbo and cuda iff CUDAPATH is not commented in user_paths_MUST_BE_EDITED_FIRST file
cudamode = $(shell cat ../../../../../user_paths_MUST_BE_EDITED_FIRST | grep -q '^CUDAPATH' && echo "1")
# $(info $(cudamode))
ifeq ($(cudamode), 1)
	# Option 1: Cuda/VBO rendering
	export CUDAPATH ?= $(shell cat ../../../../../user_paths_MUST_BE_EDITED_FIRST | grep -Po '(?<=CUDAPATH=).+')
	export THRUSTPATH  ?= $(CUDAPATH)/include
else
	# Option 2: Simple rendering
	export THRUSTPATH ?= $(shell cat ../../../../../user_paths_MUST_BE_EDITED_FIRST | grep -Po '(?<=THRUSTPATH=).+')
endif


export CUSTOM             =  default
export CUSTOMPATH_ROOT    =  $(abspath ../../../../custom)/$(CUSTOM)
export CUSTOMPATH_MODEL   =  $(CUSTOMPATH_ROOT)/model

all: main.o
	g++ -std=c++11 main.o -o genState -lboost_serialization

main.o: main.cpp
	g++ -std=c++11 -c main.cpp -o main.o -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DGRN -I$(THRUSTPATH) -I../../../../common/includes -I../../../../model/includes -I$(CUSTOMPATH_MODEL)/includes

run: all
	./genState