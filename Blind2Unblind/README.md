# Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots
[Blind2Unblind](https://arxiv.org/abs/2203.06967)

## Citing Blind2Unblind
```
@InProceedings{Wang_2022_CVPR,
    author    = {Wang, Zejin and Liu, Jiazheng and Li, Guoqing and Han, Hua},
    title     = {Blind2Unblind: Self-Supervised Image Denoising With Visible Blind Spots},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {2027-2036}
}
```
## Installation
The model is built in Python3.8.5, PyTorch 1.7.1 in Ubuntu 18.04 environment.

## Data Preparation

### 1. Prepare Training Dataset

- For processing ImageNet Validation, please run the command

  ```shell
  python ./dataset_tool.py
  ```

- For processing SIDD Medium Dataset in raw-RGB, please run the command

  ```shell
  python ./dataset_tool_raw.py
  ```

### 2. Prepare Validation Dataset

​	Please put your dataset under the path: **./Blind2Unblind/data/validation**.

## Pretrained Models
Download pre-trained models: [Google Drive](https://drive.google.com/drive/folders/1ruA6-SN1cyf30-GHS8w2YD1FG-0A-k7h?usp=sharing) 

The pre-trained models are placed in the folder: **./Blind2Unblind/pretrained_models**

```yaml
# # For synthetic denoising
# gauss25
./pretrained_models/g25_112f20_beta19.7.pth
# gauss5_50
./pretrained_models/g5-50_112rf20_beta19.4.pth
# poisson30
./pretrained_models/p30_112f20_beta19.1.pth
# poisson5_50
./pretrained_models/p5-50_112rf20_beta20.pth

# # For raw-RGB denoising
./pretrained_models/rawRGB_112rf20_beta19.4.pth

# # For fluorescence microscopy denoising
# Confocal_FISH
./pretrained_models/Confocal_FISH_112rf20_beta20.pth
# Confocal_MICE
./pretrained_models/Confocal_MICE_112rf20_beta19.7.pth
# TwoPhoton_MICE
./pretrained_models/TwoPhoton_MICE_112rf20_beta20.pth
```

## Train
* Train on synthetic dataset
```shell
python train_b2u.py --noisetype gauss25 --data_dir ./data/train/Imagenet_val --val_dirs ./data/validation --save_model_path ../experiments/results --log_name b2u_unet_gauss25_112rf20 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
```
* Train on SIDD raw-RGB Medium dataset
```shell
python train_sidd_b2u.py --data_dir ./data/train/SIDD_Medium_Raw_noisy_sub512 --val_dirs ./data/validation --save_model_path ../experiments/results --log_name b2u_unet_raw_112rf20 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
```
* Train on FMDD dataset
```shell
python train_fmdd_b2u.py --data_dir ./dataset/fmdd_sub/train --val_dirs ./dataset/fmdd_sub/validation --subfold Confocal_FISH --save_model_path ../experiments/fmdd --log_name Confocal_FISH_b2u_unet_fmdd_112rf20 --Lambda1 1.0 --Lambda2 2.0 --increase_ratio 20.0
```

## Test

* Test on **Kodak, BSD300 and Set14**

  * For noisetype: gauss25

    ```shell
    python test_b2u.py --noisetype gauss25 --checkpoint ./pretrained_models/g25_112f20_beta19.7.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_g25_112rf20 --beta 19.7
    ```

  * For noisetype: gauss5_50

    ```shell
    python test_b2u.py --noisetype gauss5_50 --checkpoint ./pretrained_models/g5-50_112rf20_beta19.4.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_g5_50_112rf20 --beta 19.4
    ```

  * For noisetype: poisson30

    ```shell
    python test_b2u.py --noisetype poisson30 --checkpoint ./pretrained_models/p30_112f20_beta19.1.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_p30_112rf20 --beta 19.1
    ```

  * For noisetype: poisson5_50

    ```shell
    python test_b2u.py --noisetype poisson5_50 --checkpoint ./pretrained_models/p5-50_112rf20_beta20.pth --test_dirs ./data/validation --save_test_path ./test --log_name b2u_unet_p5_50_112rf20 --beta 20.0
    ```

* Test on **SIDD Validation** in raw-RGB space

```shell
python test_sidd_b2u.py --checkpoint ./pretrained_models/rawRGB_112rf20_beta19.4.pth --test_dirs ./data/validation --save_test_path ./test --log_name validation_b2u_unet_raw_112rf20 --beta 19.4
```

* Test on **SIDD Benchmark** in raw-RGB space

```shell
python benchmark_sidd_b2u.py --checkpoint ./pretrained_models/rawRGB_112rf20_beta19.4.pth --test_dirs ./data/validation --save_test_path ./test --log_name benchmark_b2u_unet_raw_112rf20 --beta 19.4
```

* Test on **FMDD Validation**

  *  For Confocal_FISH

    ```shell
    python test_fmdd_b2u.py --checkpoint ./pretrained_models/Confocal_FISH_112rf20_beta20.pth --test_dirs ./dataset/fmdd_sub/validation --subfold Confocal_FISH --save_test_path ./test --log_name Confocal_FISH_b2u_unet_fmdd_112rf20 --beta 20.0
    ```

  *  For Confocal_MICE

    ```shell
    python test_fmdd_b2u.py --checkpoint ./pretrained_models/Confocal_MICE_112rf20_beta19.7.pth --test_dirs ./dataset/fmdd_sub/validation --subfold Confocal_MICE --save_test_path ./test --log_name Confocal_MICE_b2u_unet_fmdd_112rf20 --beta 19.7
    ```

  *  For TwoPhoton_MICE

    ```shell
    python test_fmdd_b2u.py --checkpoint ./pretrained_models/TwoPhoton_MICE_112rf20_beta20.pth --test_dirs ./dataset/fmdd_sub/validation --subfold TwoPhoton_MICE --save_test_path ./test --log_name TwoPhoton_MICE_b2u_unet_fmdd_112rf20 --beta 20.0
    ```


## Test a model on new data

After training or downloading a pretrained model, if you want to test the model on a new dataset follow this procedure : 

1. Put your dataset composed of images in this folder : **./Blind2Unblind/data/validation**

2. Modify the **test_b2u.py** file by adding : 

```python
def validation_your_dataset(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images
```
Make sure this function deals correctly with the images you have put in the validation folder.

Further in the file modifiy this also : 

```python
# Validation Set
Kodak_dir = os.path.join(opt.test_dirs, "Kodak24")
BSD300_dir = os.path.join(opt.test_dirs, "BSD300")
Set14_dir = os.path.join(opt.test_dirs, "Set14")
Urban100_dir_LR = os.path.join(opt.test_dirs, "Urban100_LR_x4")
BSD100_dir_LR_x2 = os.path.join(opt.test_dirs, "BSD100_LR_x2")
#Add this line
your_dataset = os.path.join(opt.test_dirs, "your_dataset_folder_name")

valid_dict = {
    "Kodak24": validation_kodak(Kodak_dir),
    "BSD300": validation_bsd300(BSD300_dir),
    "Set14": validation_Set14(Set14_dir),
    "Urban100_LR_x4": validation_urban100(Urban100_dir_LR),
    "BSD100_LR_x2": validation_bsd100(BSD100_dir_LR_x2),
    "your_dataset_name" : validation_your_dataset(your_dataset) #Add this line
}
``` 

You also need to add the number of repeats you want the model to do on your dataset (the average metrics will be the average for each image over the repetitions) : 

```python
valid_repeat_times = {"Kodak24": 10, "BSD300": 3, "Set14": 20, "Urban100_LR_x4": 3,"BSD100_LR_x2":3, "your_dataset_name" : number_of_repetition} # Must be an integer
```

3. Run the test with the type of noise you want as described in [this part](#test).

4. The results will be in the **./test/** folder. You can compare the original, noised and denoised images thanks to the **compare_pngs.py** file .

## Train a model on new data
To train you own model on a dataset, follow this procedure (nearly the same as for testing):

1. Put your dataset composed of images in this folder : **./Blind2Unblind/data/train**

2. Modify the **train_b2u.py** file by adding : 

```python
def validation_your_dataset(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images
```
Make sure this function deals correctly with the images you have put in the validation folder.

Further in the file modifiy this also : 

```python
# Validation Set
Kodak_dir = os.path.join(opt.test_dirs, "Kodak24")
BSD300_dir = os.path.join(opt.test_dirs, "BSD300")
Set14_dir = os.path.join(opt.test_dirs, "Set14")
Urban100_dir_LR = os.path.join(opt.test_dirs, "Urban100_LR_x4")
BSD100_dir_LR_x2 = os.path.join(opt.test_dirs, "BSD100_LR_x2")
#Add this line
your_dataset = os.path.join(opt.test_dirs, "your_dataset_folder_name")

valid_dict = {
    "Kodak24": validation_kodak(Kodak_dir),
    "BSD300": validation_bsd300(BSD300_dir),
    "Set14": validation_Set14(Set14_dir),
    "Urban100_LR_x4": validation_urban100(Urban100_dir_LR),
    "BSD100_LR_x2": validation_bsd100(BSD100_dir_LR_x2),
    "your_dataset_name" : validation_your_dataset(your_dataset) #Add this line
}
``` 

You also need to add the number of repeats you want the model to do on your dataset (the average metrics will be the average for each image over the repetitions) : 

```python
valid_repeat_times = {"Kodak24": 10, "BSD300": 3, "Set14": 20, "Urban100_LR_x4": 3,"BSD100_LR_x2":3, "your_dataset_name" : number_of_repetition} # Must be an integer
```

Exemple of command line:

```shell
python train_b2u.py  --noisetype gauss25   --data_dir ./data/train/OpenImages_train   --val_dirs ./data/validation   --save_model_path ./experiments/results   --device cuda   --log_name b2u_unet_OpenImages_gauss25_3  --n_epoch 50   --Lambda1 1.0   --Lambda2 2.0   --increase_ratio 20.0
```