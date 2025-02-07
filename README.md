# POGS 
## (Persistent Object Gaussian Splat for Tracking Human and Robot Manipulation of Irregularly Shaped Objects)

This is the official implementation for [POGS](https://berkeleyautomation.github.io/POGS/).

<div align="center">

[[Website]](https://berkeleyautomation.github.io/POGS/)
<!-- [[PDF]](https://autolab.berkeley.edu/assets/publications/media/2024_IROS_LEGS_CR.pdf) -->
<!-- [[Arxiv]](https://arxiv.org/abs/2409.18108) -->

<!-- insert figure -->
[![POGS Teaser](media/POGS_teaser.gif)]

</div>

Tested on Python 3.10, cuda 11.8, using conda. 

## Installation
1. Create conda environment and install relevant packages
```
conda create --name pogs -y python=3.10
conda activate pogs
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

2. [`cuml`](https://docs.rapids.ai/install) is required (for global clustering).
The best way to install it is with pip: `pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==24.2.* cuml-cu11==24.2.*`.

3. Install POGS!
```
git clone https://github.com/BerkeleyAutomation/pogs.git --recurse-submodules
cd pogs
python -m pip install -e .

ns-install-cli
```
