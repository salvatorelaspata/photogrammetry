# This could also be another Ubuntu or Debian based distribution
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Open3D system dependencies and pip
RUN apt-get update && apt-get install --no-install-recommends -y \
  libegl1 \
  libgl1 \
  libgomp1 \
  python3-pip \  
  libopencv-dev \
  python3-opencv \
  && rm -rf /var/lib/apt/lists/*

COPY ./src /source/src
COPY ./requirements.txt /source/requirements.txt

WORKDIR /source

RUN mkdir -p /source/output

RUN python3 src/_text.py > /source/src/log.txt

RUN pip install --no-cache-dir -r requirements.txt 

RUN mkdir -p /source/input/HEIC /source/output/HEIC_TO_JPG

RUN python3 /source/src/heic_to_jpg.py /source/input/HEIC /source/output/HEIC_TO_JPG

# RUN pip install --no-cache-dir -r requirements_depth.txt

RUN mkdir -p /source/output/DEPTH_MAP

RUN python3 src/depth_map.py /source/output/HEIC_TO_JPG /source/output/DEPTH_MAP