# GPGPU-Sim build/runtime environment
#   - Ubuntu 22.04, CUDA 11.8 toolkit
#   - GCC/G++ 9 (default compiler)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc-9 \
        g++-9 \
        bison \
        flex \
        zlib1g-dev \
        libglu1-mesa-dev \
        xutils-dev \
        python3 \
        python3-pip \
        git \
        wget \
        vim \
        gdb \
        doxygen \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

# Use GCC/G++ 9 as the default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

ENV GPGPUSIM_ROOT=/root/gpgpu-sim_distribution
COPY gpgpu-sim_distribution ${GPGPUSIM_ROOT}

# Build GPGPU-Sim
WORKDIR ${GPGPUSIM_ROOT}
RUN /bin/bash -c "source setup_environment && make -j$(nproc)"

# Load the GPGPU-Sim environment in every interactive shell
RUN echo "source ${GPGPUSIM_ROOT}/setup_environment > /dev/null" >> /root/.bashrc

WORKDIR /root
CMD ["/bin/bash"]
