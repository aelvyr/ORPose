# ORPose
 Pose Estimation Framework and Pipeline for detecting hand and body poses in a surgical environment.

## Installation Instructions

Necessary dependencies: CUDA toolkit

### Environment
Set up a virtual environment for running the application.
Here you should make sure to install all the packages outlined below.

The suggested package manager is Miniconda. 
You can follow the following steps:

**Step 1** Install Miniconda from the [official website](https://docs.anaconda.com/miniconda/).

**Step 2** Create a new environment:
```bash
conda create -n orpose python=3.11 -y
conda activate orpose
```
**Step 3** Install the [prerequisites](#prerequisites) (See below).

**Step 4** Install other required packages from the env file (requirements.txt):
```bash
conda install -r requirements.txt
```

The code was developed on both Windows and MacOS with M1 chip.

### Prerequisites

#### pytorch
Follow the instructions on the official [pytorch page](https://pytorch.org/). 
(This code was developed using pytorch 2.5.1)

Make sure to have CUDA drivers installed if NVIDIA GPUs are available.
You find more information on the [official website](https://developer.nvidia.com/cuda-downloads).

#### Ninja
Install ninja using the instructions in the [official website](https://ninja-build.org/).

#### mmpose
Follow the installation instructions on the [mmpose website](https://mmpose.readthedocs.io/en/latest/installation.html).
(The code was developed using mmpose v1.3.0)

#### SAM2

Install SAMv2 following the official instructions on the [sam2 github page](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation)