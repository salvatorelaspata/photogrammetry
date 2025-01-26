# This could also be another Ubuntu or Debian based distribution
FROM ubuntu:22.04

# Install Open3D system dependencies and pip
RUN apt-get update && apt-get install --no-install-recommends -y \
  libegl1 \
  libgl1 \
  libgomp1 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

COPY . /source
COPY ./requirements_heic.txt /source/requirements_heic.txt
COPY ./requirements_heic.txt /source/requirements_depth.txt
COPY ./requirements_heic.txt /source/requirements_open3d.txt

WORKDIR /source

RUN python3 src/text.py

# Installa le dipendenze Python
# RUN pip install --no-cache-dir -r requirements_heic.txt

# RUN python3 src/heic_to_jpg.py /source/input/HEIC /source/output/HEIC_TO_JPG

# RUN pip install --no-cache-dir -r requirements_depth.txt

# RUN python3 src/depth_map.py /source/output/HEIC_TO_JPG /source/output/DEPTH_MAP