# Image Denoising with Blind2Unblind

This repository contains an implementation of the Blind2Unblind image denoising framework, adapted from the paper [Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots](https://arxiv.org/abs/2203.06967) (CVPR 2022).

## Project Overview

The implementation focuses on self-supervised image denoising techniques, particularly using the Blind2Unblind method which employs a novel approach with visible blind spots. The code allows for training and testing on various datasets including standard benchmarks and custom fluorescence microscopy data.

## Repository Structure

This repository contains:

- Our literature review of denoising methods in the [méthodes](./méthodes) directory
- Implementation and usage instructions for our chosen denoising method in [Blind2Unblind](./Blind2Unblind)

## Key Features

- **Self-supervised learning**: Train denoising models without paired clean/noisy images
- **Multiple noise types support**: Gaussian, Poisson, and real-world noise handling
- **Microscopy image denoising**: Special implementation for fluorescence microscopy datasets (FMDD)
- **Flexible architecture**: UNet-based architecture that can be adapted to different image types

## Methods

The Blind2Unblind approach works by:

1. Creating artificial blind spots in noisy images
2. Training a network to reconstruct these blind spots
3. Leveraging self-supervision to learn denoising without clean targets
4. Using a specialized masking strategy with interpolation for optimal reconstruction

## Datasets

The implementation supports several datasets:
- General image datasets: Kodak24, BSD300, Set14, Urban100, BSD100
- Microscopy datasets: Confocal_FISH, Confocal_MICE, TwoPhoton_MICE
- Smartphone Image Denoising Dataset (SIDD)
- Custom datasets can be easily integrated


## Results
The model demonstrates strong performance on various benchmark datasets and can effectively denoise fluorescence microscopy images without requiring clean training data, which is particularly valuable for scientific imaging applications.

## Extending to New Data
The repository includes detailed instructions for:

Adding custom validation datasets
Training new models on custom data
Testing pretrained models on new datasets

Stevan Le Stanc