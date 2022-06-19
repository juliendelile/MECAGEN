# Build container:
# docker build . -t mecagen_default

# Run
# docker run -it --rm \
# 	-e DISPLAY=${DISPLAY} \
# 	--device /dev/dri/card0 \
# 	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
# 	mecagen_default

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt update && apt install -y \
	build-essential \
	freeglut3-dev \
	git \
	libboost-random-dev \
	libboost-serialization-dev \
	libglew-dev \
	libsdl2-dev \
	qt5-default \
	wget

WORKDIR /home

RUN wget https://github.com/NVIDIA/thrust/archive/refs/tags/1.7.0.tar.gz \
	&& tar -xf 1.7.0.tar.gz \
	&& rm 1.7.0.tar.gz

RUN git clone https://github.com/juliendelile/MECAGEN.git \
	&& cd MECAGEN \
	&& git reset --hard 76958a8

# We compile the default and the zebrafish version
# in two separate folders
RUN cp -r MECAGEN MECAGEN_zebra

RUN cd MECAGEN \
	&& make CUSTOM=default -j4 \
	&& cd mecagen \
	&& ./generate_input_files.sh default


RUN cd MECAGEN_zebra \
	&& make CUSTOM=zebrafish -j4 \
	&& cd mecagen \
	&& ./generate_input_files.sh zebrafish

# Replace the path of zebrafish demo to run from the zebra folder
RUN cd MECAGEN/mecagen \
	&& sed -i \
	"25s/.\/launchGUI.sh/cd ..\/..\/MECAGEN_zebra\/mecagen \&\& .\/launchGUI.sh/" \
	run_examples.sh 

CMD cd MECAGEN/mecagen && ./run_examples.sh
