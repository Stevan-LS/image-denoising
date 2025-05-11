# Blind2Unblind: Self-Supervised Image Denoising

This repository provides an implementation of the Blind2Unblind image denoising framework, based on the paper [Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots](https://arxiv.org/abs/2203.06967) (CVPR 2022).

## Overview
Blind2Unblind is a self-supervised image denoising method that leverages visible blind spots and masking strategies to train deep networks without requiring clean/noisy image pairs. This approach is especially effective for scientific and microscopy imaging, where clean data is scarce.

## Features
- **Self-supervised learning**: No need for paired clean/noisy images
- **Multiple noise types**: Gaussian, Poisson, and real-world noise
- **Microscopy support**: Special handling for fluorescence microscopy datasets (FMDD)
- **Flexible UNet-based architecture**
- **Easy extension to new datasets**

## Repository Structure
- Literature review: [`méthodes/`](./méthodes)
- Blind2Unblind implementation and scripts: [`Blind2Unblind/`](./Blind2Unblind)
- Example tests and experiments: [`test/`](./test)

## Quickstart

### 1. Installation

```bash
cd Blind2Unblind
pip install -r requirements.txt
```

### 2. Prepare Data
- See [Blind2Unblind/README.md](./Blind2Unblind/README.md#data-preparation) for dataset preparation scripts and folder structure.

### 3. Download Pretrained Models (Optional)
- Pretrained models are available [here](https://drive.google.com/drive/folders/1ruA6-SN1cyf30-GHS8w2YD1FG-0A-k7h?usp=sharing). Place them in `Blind2Unblind/pretrained_models/`.

### 4. Training Example
```bash
python Blind2Unblind/train_b2u.py --noisetype gauss25 --data_dir ./Blind2Unblind/data/train/Imagenet_val --val_dirs ./Blind2Unblind/data/validation --save_model_path ./experiments/results --log_name b2u_unet_gauss25 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
```

### 5. Testing Example
```bash
python Blind2Unblind/test_b2u.py --noisetype gauss25 --checkpoint ./Blind2Unblind/pretrained_models/g25_112f20_beta19.7.pth --test_dirs ./Blind2Unblind/data/validation --save_test_path ./test --log_name b2u_unet_g25 --beta 19.7
```

For more training and testing options (including SIDD and FMDD datasets), see [Blind2Unblind/README.md](./Blind2Unblind/README.md#train) and [#test].

## Supported Datasets
- General: Kodak24, BSD300, Set14, Urban100, BSD100
- Microscopy: Confocal_FISH, Confocal_MICE, TwoPhoton_MICE
- Smartphone: SIDD
- Custom datasets: Easily integrated (see [Blind2Unblind/README.md](./Blind2Unblind/README.md#test-a-model-on-new-data))

## Results
Blind2Unblind achieves strong denoising performance on standard benchmarks and excels in microscopy image denoising without clean training data.

## Extending to New Data
- Add your own datasets for validation or training by following the instructions in [Blind2Unblind/README.md](./Blind2Unblind/README.md#test-a-model-on-new-data) and [#train-a-model-on-new-data].


## Acknowledgments
- This repository adapts and extends the official Blind2Unblind implementation.
- For detailed usage, advanced options, and troubleshooting, see [Blind2Unblind/README.md](./Blind2Unblind/README.md).

---
**Author:** Stevan Le Stanc