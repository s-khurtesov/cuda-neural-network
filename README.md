# cuda-neural-network
Deep neural network. Uses CUDA with cuDNN and cuBLAS_v2 libraries. Provides flexible model building. As an example, classificates cell Images for detecting malaria.

Prerequisites
=============

* MS Visual Studio (tested on VS 2019): https://visualstudio.microsoft.com/ru/vs/
* CUDA Toolkit (tested on CUDA v11.0.1): https://developer.nvidia.com/cuda-downloads
* cuDNN - GPU-accelerated library of primitives for deep neural networks (tested on cuDNN v8 for CUDA v11): https://developer.nvidia.com/cuDNN/
* Dataset: https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria

Building
===========

The project can be oppened and builded with Visual Studio (MSBuild). Recommended to use Debug configuration.

Running
=======
Output files are stored at x64/Release folder (or x64/Debug). Provided example uses images of cells resized to 50x50px RGB.
