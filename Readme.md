# Docker Environment

The provided `Dockerfile` builds an image (Ubuntu 22.04, CUDA 11.8, GCC 9) with
`gpgpu-sim_distribution` already compiled and ready to run.

## 1. Clone the repository

`LLMServingSim` is included as a git submodule, so clone with `--recursive`:

```bash
git clone --recursive https://github.com/comsys-lab/Ariadne.git
cd Ariadne
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## 2. Build the image

Run from the repository root (the directory that contains the `Dockerfile`):

```bash
docker build -t gpgpu-sim .
```

## 3. Run a container

```bash
docker run -it --name gpgpu-sim gpgpu-sim
```

To work on files from the host inside the container, mount a directory:

```bash
docker run -it --name gpgpu-sim -v $(pwd):/workspace gpgpu-sim
```

GPGPU-Sim simulates the GPU in software, so you do not need a physical GPU or
`--gpus`.

## 4. Inside the container

GPGPU-Sim is located at `/root/gpgpu-sim_distribution`. Every shell sources its
`setup_environment` automatically, so CUDA applications that link the CUDA
runtime dynamically (`nvcc --cudart shared`) run on the simulator. Copy a `gpgpusim.config`
(e.g., from `/root/gpgpu-sim_distribution/configs/tested-cfgs/`) into the
directory where you run the application.

To rebuild GPGPU-Sim after you modify its source:

```bash
cd /root/gpgpu-sim_distribution
make -j$(nproc)
```
