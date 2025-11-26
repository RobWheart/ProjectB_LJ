# ProjectB_LJ
Code files for Data Science Research Project B

```
============================================================
                ENVIRONMENT INFORMATION
============================================================
Python Version: 3.9.21
OS: Linux 6.8.0-52-generic
Current Working Directory: /home/ikning/Research_Project_B

----- PyTorch -----
PyTorch Version: 2.8.0+cu128
CUDA Available: True
CUDA Version (PyTorch compiled): 12.8
cuDNN Version: 91002
GPU Device Count: 1
GPU Name: NVIDIA GeForce RTX 2080 Ti

PyTorch Deterministic Flags:
cudnn.deterministic: False
cudnn.benchmark: False

----- Torchvision -----
torchvision Version: 0.23.0+cu128

----- NumPy -----
NumPy Version: 2.0.2

----- MONAI -----
MONAI Version: 1.5.0
============================================================
```


# Code Structure

This directory contains the following core scripts and modules:

Unet.py
Implements the 3D U-Net architecture used in all U-Net–based experiments.

data_maker.py
Generates the simulated MRI dataset, including hippocampal structures, localized deformations, partial-volume synthesis, and voxel-wise difference maps.

data_loader.py
Builds the dataset loaders for training, validation, and testing.
Ensures correct dataset splits and reproducible sampling.

train_unet.py
Trains and validates the 3D U-Net model on the simulated dataset using the WMSE loss function.

train_unet_coarse_fine.py
Trains the 3D U-Net model using the coarse-to-fine training strategy designed to improve sensitivity to small structural changes.

train_swin.py
Trains and validates the SwinUNETR model on the simulated dataset using the WMSE loss function.

train_swin_coarse_fine.py
Trains the SwinUNETR model using the coarse-to-fine training strategy on the same simulated dataset.

All random processes are controlled by fixed random seeds to ensure full reproducibility.
Please keep the relative path structure unchanged when running the scripts to ensure correct data loading and model checkpoint saving.



