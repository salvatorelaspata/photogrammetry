FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install apt-getable dependencies
RUN apt-get update \
  && apt-get install -y \
  build-essential \
  cmake \
  git \
  libeigen3-dev \
  libopencv-dev \
  libceres-dev \
  python3-dev \
  python3-numpy \
  python3-opencv \
  python3-pip \
  python3-pyproj \
  python3-scipy \
  python3-yaml \
  curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Clone OpenSfM repository
RUN git clone --recursive https://github.com/mapillary/OpenSfM /source/OpenSfM

RUN git clone --recursive https://github.com/mapillary/OpenSfM /source/OpenSfMFix

# COPY . /source/OpenSfM

WORKDIR /source/OpenSfM
# In the root of the main repo, run:
#   `git submodule update --init --recursive
#   This will load the removed submodules.
RUN git submodule update --init --recursive

# In the main repo, remove pyyaml==5.4.1 from requirements.txt and install it manually with the following command:
#  pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.1
RUN pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.1

WORKDIR /source/OpenSfMFix

# In the second repo, go to the August 9 commit:
#  git checkout ef872b2399cf1cc036d91e950f210a7be33c2745
RUN git checkout ef872b2399cf1cc036d91e950f210a7be33c2745

# From this commit, copy the missing files (gmock_gtest_all.cc, gmock_main.cc, testing_main.cc, and any other required files) into the main repository.
# Now you should be able to build the image from the modified dockerfile in which you should install ceres 2.2 insted ceres 2.0

# Create directory for ceres-solver
RUN mkdir -p /source/OpenSfM/third_party/ceres-solver/internal/ceres/
# RUN mkdir -p /source/OpenSfM/third_party/gtest/

# Copy the missing files
RUN cp opensfm/src/third_party/gtest/gmock_gtest_all.cc /source/OpenSfM/third_party/ceres-solver/internal/ceres/
# RUN cp opensfm/src/third_party/gtest/gmock_gtest_all.cc /source/OpenSfM/third_party/gtest/

RUN cp opensfm/src/third_party/gtest/gmock_main.cc /source/OpenSfM/third_party/ceres-solver/internal/ceres/
# RUN cp opensfm/src/third_party/gtest/gmock_main.cc /source/OpenSfM/third_party/gtest/

RUN cp opensfm/src/testing_main.cc /source/OpenSfM/third_party/ceres-solver/internal/ceres/
# RUN cp opensfm/src/testing_main.cc /source/OpenSfM/third_party/gtest/

# Copy vlfeat to the main repository

WORKDIR /source/OpenSfM

RUN cp -r opensfm/src/third_party/vlfeat /source/OpenSfM/third_party/vlfeat/
# CMake
# RUN apt-get install cmake
# RUN curl -L http://ceres-solver.org/ceres-solver-2.2.0.tar.gz | tar xz && \ 
#   mkdir ceres-bin && cd ceres-bin && \ 
#   cmake ../ceres-solver-2.2.0 && \ 
#   make -j3 && \ 
#   make install

RUN pip3 install -r requirements.txt && \
  python3 setup.py build