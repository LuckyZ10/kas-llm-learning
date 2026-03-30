# DFT+LAMMPS Framework - Docker 镜像
# 完整运行环境：VASP + Quantum ESPRESSO + LAMMPS + DeepMD + NEP

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

LABEL maintainer="DFT+LAMMPS Framework Team"
LABEL description="Complete DFT/MD/ML environment for multi-scale materials simulation"
LABEL version="1.0.0"

# 防止交互式配置提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# =============================================================================
# 基础系统依赖
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 编译工具
    build-essential \
    gfortran \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    # Python环境
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    # 数值计算库
    libopenblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libfftw3-mpi-dev \
    libscalapack-mpi-dev \
    # MPI支持
    openmpi-bin \
    libopenmpi-dev \
    # 其他依赖
    libhdf5-dev \
    libpng-dev \
    libfreetype6-dev \
    libxft-dev \
    libxml2-dev \
    libxslt1-dev \
    libz-dev \
    vim \
    nano \
    ssh \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# 设置Python3.10为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 升级pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Python科学计算环境
# =============================================================================
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    scikit-learn==1.3.0 \
    scikit-image==0.21.0 \
    plotly==5.15.0 \
    jupyter==1.0.0 \
    jupyterlab==4.0.5 \
    ipywidgets==8.1.0 \
    tqdm==4.65.0 \
    pyyaml==6.0.1

# =============================================================================
# ASE和Pymatgen
# =============================================================================
RUN pip3 install --no-cache-dir \
    ase==3.22.1 \
    pymatgen==2023.7.17 \
    mp-api==0.33.3

# =============================================================================
# LAMMPS (GPU支持)
# =============================================================================
WORKDIR /opt
RUN git clone --depth 1 --branch stable_2Aug2023_update3 https://github.com/lammps/lammps.git lammps

WORKDIR /opt/lammps/build
RUN cmake ../cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/opt/lammps \
    -D CMAKE_CXX_COMPILER=g++ \
    -D BUILD_MPI=yes \
    -D BUILD_OMP=yes \
    -D PKG_MOLECULE=yes \
    -D PKG_KSPACE=yes \
    -D PKG_MANYBODY=yes \
    -D PKG_MISC=yes \
    -D PKG_GPU=yes \
    -D GPU_API=cuda \
    -D GPU_ARCH=sm_70 \
    -D PKG_DEEPMD=yes \
    -D PKG_PYTHON=yes \
    && make -j$(nproc) \
    && make install

ENV PATH="/opt/lammps/bin:${PATH}"
ENV LAMMPS_POTENTIALS="/opt/lammps/share/lammps/potentials"

# =============================================================================
# DeepMD-kit (v2.2.4)
# =============================================================================
RUN pip3 install --no-cache-dir \
    tensorflow==2.12.0 \
    deepmd-kit==2.2.4 \
    dpgen==0.11.0

# 配置DeepMD环境
ENV DEEPMD_ROOT=/usr/local
ENV PATH="${DEEPMD_ROOT}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${DEEPMD_ROOT}/lib:${LD_LIBRARY_PATH}"

# =============================================================================
# NEP (GPUMD)
# =============================================================================
WORKDIR /opt
RUN git clone --depth 1 https://github.com/brucefan1983/GPUMD.git gpumd

WORKDIR /opt/gpumd/src
RUN make -j$(nproc) \
    && cp nep /usr/local/bin/ \
    && cp gpumd /usr/local/bin/

# =============================================================================
# VASP (需要用户自行提供)
# 注意: VASP是商业软件，需要许可证。用户应将vasp二进制文件挂载到容器中
# =============================================================================
# 创建VASP目录结构
RUN mkdir -p /opt/vasp/bin /opt/vasp/potentials
ENV VASP_HOME=/opt/vasp
ENV PATH="${VASP_HOME}/bin:${PATH}"

# 注意: 用户需要在运行时将vasp_std和vasp_gam挂载到/opt/vasp/bin/

# =============================================================================
# Quantum ESPRESSO
# =============================================================================
WORKDIR /opt
RUN wget -q https://github.com/QEF/q-e/releases/download/qe-7.2/qe-7.2-ReleasePack.tgz \
    && tar xzf qe-7.2-ReleasePack.tgz \
    && rm qe-7.2-ReleasePack.tgz

WORKDIR /opt/qe-7.2
RUN ./configure --with-scalapack=yes \
    && make -j$(nproc) all \
    && make install

ENV PATH="/opt/qe-7.2/bin:${PATH}"
ENV ESPRESSO_PSEUDO=/opt/qe-7.2/pseudo

# =============================================================================
# MACE (可选ML势)
# =============================================================================
RUN pip3 install --no-cache-dir \
    mace-torch==0.3.0 \
    e3nn==0.5.1 \
    torch==2.0.1 \
    torch-geometric==2.3.1

# =============================================================================
# OVITO (可视化)
# =============================================================================
WORKDIR /opt
RUN wget -q https://www.ovito.org/download/master/ovito-basic-3.9.4-x86_64.tar.xz \
    && tar -xf ovito-basic-3.9.4-x86_64.tar.xz \
    && mv ovito-basic-3.9.4-x86_64 ovito \
    && rm ovito-basic-3.9.4-x86_64.tar.xz

ENV PATH="/opt/ovito/bin:${PATH}"

# =============================================================================
# VMD (可视化)
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    vmd \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# DFT+LAMMPS Framework
# =============================================================================
WORKDIR /workspace
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . /workspace/dft_lammps_research/

# 设置环境变量
ENV PYTHONPATH="/workspace/dft_lammps_research:${PYTHONPATH}"
ENV DFT_LAMMPS_HOME=/workspace/dft_lammps_research

# =============================================================================
# 创建非root用户
# =============================================================================
RUN useradd -m -s /bin/bash -u 1000 researcher \
    && usermod -aG sudo researcher \
    && echo "researcher:researcher" | chpasswd

# 设置工作目录权限
RUN chown -R researcher:researcher /workspace

# =============================================================================
# 启动配置
# =============================================================================
EXPOSE 8888 8050 5006

# 入口脚本
COPY docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
