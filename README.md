# POGS 
## Persistent Object Gaussian Splat for Tracking Human and Robot Manipulation of Irregularly Shaped Objects

<div align="center">

[[Website]](https://berkeleyautomation.github.io/POGS/)
<!-- [[PDF]](https://autolab.berkeley.edu/assets/publications/media/2024_IROS_LEGS_CR.pdf) -->
<!-- [[Arxiv]](https://arxiv.org/abs/2409.18108) -->

<!-- insert figure -->
<!-- ![POGS Teaser](media/POGS_teaser.gif) -->
<img src="media/POGS_teaser.gif" width="650"/>
<div style="height: 50px;">&nbsp;</div>
<img src="media/POGS_servoing.gif" width="650"/>
<div style="height: 50px;">&nbsp;</div>

<!-- ![POGS Servoing](media/POGS_servoing.gif) -->
</div>

This repository contains the official implementation for [POGS](https://berkeleyautomation.github.io/POGS/).

Tested on Python 3.10, cuda 11.8, using conda. 

## Installation
1. Create conda environment and install relevant packages
```
conda create --name pogs_env -y python=3.10
conda activate pogs_env
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install warp-lang
```

2. [`cuml`](https://docs.rapids.ai/install) is required (for global clustering).
The best way to install it is with pip: `pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==24.2.* cuml-cu11==24.2.*`.

3. Install POGS!
```
git clone https://github.com/uynitsuj/pogs.git --recurse-submodules
cd pogs
python -m pip install -e .

ns-install-cli
```

4. There is also a physical robot action component with the UR5 and Zed cameras. To install the stuff relevant for that, do the following:
### ur5py
```
pip install ur_rtde==1.4.2
pip install cowsay
pip install opt-einsum
pip install pyvista
pip install autolab-core
cd ~/pogs/pogs/dependencies/ur5py
pip install -e .
```

### RAFT-Stereo
```
cd ~/pogs/pogs/dependencies/raftstereo
bash download_models.sh
pip install -e .
```

## Usage
### Calibrate wrist mounted and third person cameras
Before training/tracking POGS, make sure wrist mounted camera and third-person view camera are calibrated. We use an Aruco marker for the calibration
```
cd ~/pogs/pogs/scripts
python calibrate_cameras.py
```

### Scene Capture
Script used to perform hemisphere capture with robot on tabletop scene. We used manual trajectory but you can also put the robot in "teach" mode to capture trajectory.
```
cd ~/pogs/pogs/scripts
python scene_capture.py --scene DATA_NAME
```