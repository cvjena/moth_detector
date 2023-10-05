# Moth Detector

## Installation

This repository requires `cupy>=7.8,<8.0`, but unfortunately, the wheel version supporting CUDA 11 (required for NVIDIA RTX 3XXX GPUs) is only compiled for python3.7 (https://pypi.org/project/cupy-cuda110/7.8.0/).
Hence, we need to install cupy from pip, but before doing so, we have to install the `cudatoolkit` package containing `nvcc`:

```bash
conda create -n moth_scanner python~=3.9.0 mpi4py cython~=0.28
conda activate moth_scanner
conda install -c conda-forge -c nvidia cudnn~=8.0.0 nccl cutensor \
	cudatoolkit~=11.0.3	cudatoolkit-dev~=11.0.3 numpy~=1.23.0
```

To be sure, check with `which nvcc` whether the CUDA-Compiler is successfully installed in your conda environment.

Now, you can install the dependencies with
```bash
pip install -r requirements.txt
```

Run the following to validate the `cupy` installation:
```bash
python -c "import cupy as cp; cp.show_config(); print(cp.zeros(8) + 2)"
```
## Usage

All scripts are located in the `scripts` folder.

