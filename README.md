# Moth Detector

Code for the paper "[Deep Learning Pipeline for Automated Visual Moth Monitoring: Insect Localization and Species Classification](https://arxiv.org/abs/2307.15427)"

*check out the [Moth-Scanner Repo](https://github.com/cvjena/moth_scanner)*

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

## Citation
You are welcome to use our code in your research! If you do so please cite it as:

```bibtex
@article{korsch2023deep,
  title={Deep learning pipeline for automated visual moth monitoring: insect localization and species classification},
  author={Korsch, Dimitri and Bodesheim, Paul and Denzler, Joachim},
  journal={arXiv preprint arXiv:2307.15427},
  year={2023}
}
```

## License
This work is licensed under a [GNU Affero General Public License][agplv3].

[![AGPLv3][agplv3-image]][agplv3]

[agplv3]: https://www.gnu.org/licenses/agpl-3.0.html
[agplv3-image]: https://www.gnu.org/graphics/agplv3-88x31.png
