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
  libtbb-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["flask", "run", "--host=0.0.0.0", "--port=5001", "--reload"]


