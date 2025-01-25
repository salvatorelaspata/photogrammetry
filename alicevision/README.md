# Alicevision (unused)

AliceVision is a Photogrammetric Computer Vision Framework which provides a 3D Reconstruction and Camera Tracking algorithms.

Meshroom is a 3D reconstruction software that uses AliceVision as its backend. It is a free and open-source software that allows you to create 3D models from a set of photographs. It is a great tool for creating 3D models of objects, buildings, and landscapes.

> **Note:** This project requires a CUDA-enabled GPU to run. 
> If you do not have a CUDA-enabled GPU, you can use the CPU version of the software, but it will be much slower.

## Pre-requisites

- Docker
- Docker Compose
- CUDA-enabled GPU

## Installation

1. Provide the images input in the `input` folder.
2. Run the following command to start the application:

```bash
docker compose up --build -d
```

3. Run the meshroom_batch command to start the 3D reconstruction process.

> **Note:** The follow command must be run from the `meshroom` container.

```bash
docker exec -it meshroom bash
meshroom_batch --input /input --output /output
```

> **Note:** Replace `input` and `output` with the appropriate paths.
