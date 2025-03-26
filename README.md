# README

## Introduction

This repository provides the artifacts for our paper titled *"MetaKernel: Enabling Efficient Encrypted Neural Network Inference Through Unified MVM and Convolution"*, including the source code of our MKR implementation and raw data.

## Source

The source code for MKR is located in the **source** directory. MKR is implemented within the open-source ANT-ACE FHE compiler. This directory includes both the ANT-ACE source code and our MKR implementation, organized as follows:

- **air-infra**: The ANT-ACE FHE compiler infrastructure.  
- **nn-addon**: The ONNX front end, including the tensor IR and vector IR. MKR is implemented here as part of the lowering process from tensor IR to vector IR.  
- **fhe-cmplr**: Contains ANT-ACE’s remaining three IRs: SIHE IR, CKKS IR, and Poly IR.


## Raw Data

The raw data is located in the **raw_data** directory, including the source code for DNN models (and their kernels) and the experimental results recorded in log files.

- **models**: The PyTorch programs for the MVM and Conv kernels, as well as the DNN models evaluated in the paper. These PyTorch programs are first converted to ONNX models, which are then compiled by ANT-ACE (where MKR is implemented) into equivalent FHE programs for encrypted DNN inference.

- **logs**: Contains the log files generated during our evaluation, which include the data used to produce the tables and figures in the paper.


## Test Cases

### Small Conv Kernels

| Kernel | Model | MKR timing log | MKR statistics log | FHELIPE timing log | FHELIPE statistics log |
|--------|-------|----------------|--------------------|--------------------|------------------------|
|Conv1   |conv_3x16x32x3.py  |mkr_conv_3x16x32x3.run.log  |mkr_conv_3x16x32x3.sts.log  |fhelipe_conv_3x16x32x3.run.log  |fhelipe_conv_3x16x32x3.sts.log  |
|Conv2   |conv_16x16x32x3.py |mkr_conv_16x16x32x3.run.log |mkr_conv_16x16x32x3.sts.log |fhelipe_conv_16x16x32x3.run.log |fhelipe_conv_16x16x32x3.sts.log |
|Conv3   |conv_32x32x16x3.py |mkr_conv_32x32x16x3.run.log |mkr_conv_32x32x16x3.sts.log |fhelipe_conv_32x32x16x3.run.log |fhelipe_conv_32x32x16x3.sts.log |
|Conv4   |conv_64x64x8x3.py  |mkr_conv_64x64x8x3.run.log  |mkr_conv_64x64x8x3.sts.log  |fhelipe_conv_64x64x8x3.run.log  |fhelipe_conv_64x64x8x3.sts.log  |

### Large Conv Kernels

| Kernel | Model | MKR timing log | MKR statistics log | FHELIPE timing log | FHELIPE statistics log |
|--------|-------|----------------|--------------------|--------------------|------------------------|
|Conv1_large |conv_64x64x56x3.py   |mkr_conv_64x64x56x3.run.log   |mkr_conv_64x64x56x3.sts.log   |fhelipe_conv_64x64x56x3.run.log   |fhelipe_conv_64x64x56x3.sts.log   |
|Conv2_large |conv_128x128x28x3.py |mkr_conv_128x128x28x3.run.log |mkr_conv_128x128x28x3.sts.log |fhelipe_conv_128x128x28x3.run.log |fhelipe_conv_128x128x28x3.sts.log |
|Conv3_large |conv_256x256x14x3.py |mkr_conv_256x256x14x3.run.log |mkr_conv_256x256x14x3.sts.log |fhelipe_conv_256x256x14x3.run.log |fhelipe_conv_256x256x14x3.sts.log |
|Conv4_large |conv_512x512x7x3.py  |mkr_conv_512x512x7x3.run.log  |mkr_conv_512x512x7x3.sts.log  |fhelipe_conv_512x512x7x3.run.log  |fhelipe_conv_512x512x7x3.sts.log  |

### MVM Kernels

| Kernel | Model | MKR timing log | MKR statistics log | FHELIPE timing log | FHELIPE statistics log |
|--------|-------|----------------|--------------------|--------------------|------------------------|
|MVM1    |gemv_4096x4096.py  |mkr_gemv_4096x4096.run.log  |mkr_gemv_4096x4096.sts.log  |fhelipe_fc_4096x4096.run.log  |fhelipe_fc_4096x4096.sts.log |
|MVM2    |gemv_4096x25088.py |mkr_gemv_4096x25088.run.log |mkr_gemv_4096x25088.sts.log |fhelipe_fc_4096x25088.run.log | N/A |

### DNN Models on CIFAR 

| Kernel | Model | MKR timing log | MKR statistics log | FHELIPE timing log | FHELIPE statistics log |
|--------|-------|----------------|--------------------|--------------------|------------------------|
|ResNet20   |resnet20_cifar10.py    |mkr_resnet20_cifar.run.log   |mkr_resnet20_cifar.sts.log   |fhelipe_resnet20_cifar.run.log   |fhelipe_resnet20_cifar.sts.log   |
|SqueezeNet |squeezenet_cifar10.py  |mkr_squeezenet_cifar.run.log |mkr_squeezenet_cifar.sts.log |fhelipe_squeezenet_cifar.run.log |fhelipe_squeezenet_cifar.sts.log |
|AlexNet    |alexnet_cifar10.py     |mkr_alexnet_cifar.run.log    |mkr_alexnet_cifar.sts.log    |fhelipe_alexnet_cifar.run.log    |fhelipe_alexnet_cifar.sts.log    |
|VGG11      |vgg11_cifar10.py       |mkr_vgg11_cifar.run.log      |mkr_vgg11_cifar.sts.log      |fhelipe_vgg11_cifar.run.log      |fhelipe_vgg11_cifar.sts.log      |
|MobileNet  |mobilenet_cifar10.py   |mkr_mobilenet_cifar.run.log  |mkr_mobilenet_cifar.sts.log  | N/A | N/A |

### DNN models on ImageNet 

| Kernel | Model | MKR timing log | MKR statistics log |
|--------|-------|----------------|--------------------|
|ResNet18   |resnet18_imagenet.py   |mkr_resnet18_imagenet.run.log   |mkr_resnet18_imagenet.sts.log   |
|SqueezeNet |squeezenet_imagenet.py |mkr_squeezenet_imagenet.run.log |mkr_squeezenet_imagenet.sts.log |
|AlexNet    |alexnet_imagenet.py    |mkr_alexnet_imagenet.run.log    |mkr_alexnet_imagenet.sts.log    |
|VGG11      |vgg11_imagenet.py      |mkr_vgg11_imagenet.run.log      |mkr_vgg11_imagenet.sts.log      |
|MobileNet  |mobilenet_imagenet.py  |mkr_mobilenet_imagenet.run.log  |mkr_mobilenet_imagenet.sts.log  |

