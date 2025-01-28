FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

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

COPY ./entrypoint.sh /source/entrypoint.sh

RUN chmod +x /source/entrypoint.sh

CMD ["/source/entrypoint.sh"]