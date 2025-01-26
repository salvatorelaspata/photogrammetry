# This could also be another Ubuntu or Debian based distribution
FROM ubuntu:22.04

# Install Open3D system dependencies and pip
RUN apt-get update && apt-get install --no-install-recommends -y \
  libegl1 \
  libgl1 \
  libgomp1 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

COPY . /source/

WORKDIR /source

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando per eseguire lo script Python
#CMD ["python3", "test.py", "/source/input", "/source/output"]
CMD ["python3", "test2.py", "/source/input", "/source/output/test.ply"]