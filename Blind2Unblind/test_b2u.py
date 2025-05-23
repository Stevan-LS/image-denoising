from __future__ import division
import os
import logging
import time
import glob
import datetime
import argparse
import numpy as np
from scipy.io import loadmat, savemat
import random

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from arch_unet import UNet
import utils as util
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25", choices=['gauss25', 'gauss5_50', 'poisson30', 'poisson5_50','forge'])
parser.add_argument('--checkpoint', type=str, default='./*.pth')
parser.add_argument('--test_dirs', type=str, default='./data/validation')
parser.add_argument('--save_test_path', type=str, default='./test')
parser.add_argument('--log_name', type=str, default='b2u_unet_g25_112rf20')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument("--beta", type=float, default=20.0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
torch.set_num_threads(8)

# config loggers. Before it, the log will not work
os.makedirs(opt.save_test_path, exist_ok=True)
util.setup_logger(
    "test",
    opt.save_test_path,
    "test_" + opt.log_name,
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("test")

def save_network(network, epoch, name):
    save_path = os.path.join(opt.save_path, opt.log_name, systime)
    os.makedirs(save_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_path = os.path.join(save_path, model_name)
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
    logger.info('Checkpoint saved to {}'.format(save_path))


def load_network(load_path, network, strict=True):
    assert load_path is not None
    logger.info("Loading model from [{:s}] ...".format(load_path))
    if isinstance(network, nn.DataParallel) or isinstance(
        network, nn.parallel.DistributedDataParallel
    ):
        network = network.module
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith("module."):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)
    return network


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"
        elif style.startswith('forge'):
            self.style = "forge"
            # Extraction des paramètres s'ils sont fournis
            params_str = style.replace('forge', '')
            if params_str:
                self.params = [float(p) for p in params_str.split('_')]
                if len(self.params) == 3:
                    self.poussiere_intensity = self.params[0]
                    self.calamine_spots = int(self.params[1])
                    self.grain_strength = self.params[2]
                else:
                    # Valeurs par défaut si les paramètres sont incorrects
                    self.poussiere_intensity = 0.02
                    self.calamine_spots = 30
                    self.grain_strength = 15
            else:
                # Valeurs par défaut
                self.poussiere_intensity = 0.02
                self.calamine_spots = 30
                self.grain_strength = 15

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "forge":
            batch_noisy = np.zeros_like(x)

            for i in range(shape[0]):
                # Work directly with RGB image
                img_255 = (x[i] * 255).astype(np.uint8)

                # Apply forge noise directly to RGB
                noisy_img = self._add_forge_noise_numpy(img_255, 
                                                       self.poussiere_intensity,
                                                       self.calamine_spots,
                                                       self.grain_strength)

                # Scale back to [0, 1]
                batch_noisy[i] = noisy_img.astype(np.float32) / 255.0

            return batch_noisy

    def add_valid_noise(self, x):
        shape = x.shape
        print(shape)
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "forge":
            batch_noisy = np.zeros_like(x)
            x = (x * 255).astype(np.uint8)
            noisy_img = self._add_forge_noise_numpy(x, 
                                                       self.poussiere_intensity,
                                                       self.calamine_spots,
                                                       self.grain_strength)
                
            batch_noisy = noisy_img.astype(np.float32) / 255.0
                
            return batch_noisy

    def _add_forge_noise_numpy(self, image, poussiere_intensity=0.02, calamine_spots=30, grain_strength=15):
        """
        Add forge noise to RGB image
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
        Returns:
            Noisy RGB image (H, W, 3)
        """
        # Create a copy of the image
        print(image.shape)
        noisy = image.copy().astype(np.float32)
        height, width, channels = noisy.shape

        # 1. Ajout de bruit de poussière (salt and pepper)
        poussiere_mask = np.random.random(image.shape) < poussiere_intensity
        dust_values = np.random.randint(180, 255, size=image.shape)
        noisy[poussiere_mask] = dust_values[poussiere_mask]

        # 2. Ajout de taches de calamine (taches sombres)
        for _ in range(calamine_spots):
            # Position aléatoire
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)

            # Taille aléatoire
            size = np.random.randint(5, 30)

            # Opacité aléatoire pour la calamine
            opacity = np.random.uniform(0.4, 0.9)

            # Dessiner une tache sombre avec un dégradé
            for i in range(height):
                for j in range(width):
                    distance = np.sqrt((i - y) ** 2 + (j - x) ** 2)
                    if distance < size:
                        # Plus sombre au centre, s'atténue vers les bords
                        darkness = opacity * (1 - distance / size)
                        noisy[i, j] = noisy[i, j] * (1 - darkness) + 30 * darkness

        # 3. Variation de luminosité avec un filtre gaussien
        luminosity_variation = cv2.GaussianBlur(np.random.randn(height, width, channels) * 10, (21, 21), 0)
        noisy += luminosity_variation

        # 4. Ajout de texture granuleuse
        grain = np.random.randn(height, width, channels) * grain_strength
        noisy += grain

        # Normalisation des valeurs
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        # Ajustement du contraste pour accentuer l'effet forge
        alpha = 1.1  # Facteur de contraste
        beta = -5    # Ajustement de luminosité
        noisy = cv2.convertScaleAbs(noisy, alpha=alpha, beta=beta)

        return noisy

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


# def depth_to_space(x, block_size):
#     """
#     Input: (N, C × ∏(kernel_size), L)
#     Output: (N, C, output_size[0], output_size[1], ...)
#     """
#     n, c, h, w = x.size()
#     x = x.reshape(n, c, h * w)
#     folded_x = torch.nn.functional.fold(
#         input=x, output_size=(h*block_size, w*block_size), kernel_size=block_size, stride=block_size)
#     return folded_x


def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


def generate_mask(img, width=4, mask_type='random'):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                       dtype=torch.int64,
                       device=img.device)
    idx_list = torch.arange(
        0, width**2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n * h // width * w // width, ),
                         dtype=torch.int64,
                         device=img.device)

    if mask_type == 'random':
        torch.randint(low=0,
                      high=len(idx_list),
                      size=(n * h // width * w // width, ),
                      device=img.device,
                      generator=get_generator(device=img.device),
                      out=rd_idx)
    elif mask_type == 'batch':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(n, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(h // width * w // width)
    elif mask_type == 'all':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(1, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(n * h // width * w // width)
    elif 'fix' in mask_type:
        index = mask_type.split('_')[-1]
        index = torch.from_numpy(np.array(index).astype(
            np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // width * w // width * width**2,
                                step=width**2,
                                dtype=torch.int64,
                                device=img.device)

    mask[rd_pair_idx] = 1

    mask = depth_to_space(mask.type_as(img).view(
        n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

    return mask


def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv


class Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n,self.width**2,c,h,w), device=img.device)
        masks = torch.zeros((n,self.width**2,1,h,w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:,i,...] = x
            masks[:,i,...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


class DataLoader_SIDD_Medium_Raw(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_SIDD_Medium_Raw, self).__init__()
        self.data_dir = data_dir
        # get images path
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = loadmat(fn)["x"]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def get_SIDD_validation(dataset_dir):
    val_data_dict = loadmat(
        os.path.join(dataset_dir, "ValidationNoisyBlocksRaw.mat"))
    val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']
    val_data_dict = loadmat(
        os.path.join(dataset_dir, 'ValidationGtBlocksRaw.mat'))
    val_data_gt = val_data_dict['ValidationGtBlocksRaw']
    num_img, num_block, _, _ = val_data_gt.shape
    return num_img, num_block, val_data_noisy, val_data_gt


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images

def validation_urban100(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images

def validation_bsd100(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images

def validation_bsd500_test(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref, data_range=255.0):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr

# Validation Set
Kodak_dir = os.path.join(opt.test_dirs, "Kodak24")
BSD300_dir = os.path.join(opt.test_dirs, "BSD300")
Set14_dir = os.path.join(opt.test_dirs, "Set14")
Urban100_dir_LR = os.path.join(opt.test_dirs, "Urban100_LR_x4")
BSD100_dir_LR_x2 = os.path.join(opt.test_dirs, "BSD100_LR_x2")
BSD500_test_dir = os.path.join(opt.test_dirs, "BSD_500_test")
#Urban100_dir_HR = os.path.join(opt.test_dirs, "Urban100_HR")

valid_dict = {
    #"Kodak24": validation_kodak(Kodak_dir),
    #"BSD300": validation_bsd300(BSD300_dir),
    #"Set14": validation_Set14(Set14_dir),
    "Urban100_LR_x4": validation_urban100(Urban100_dir_LR),
    "BSD100_LR_x2": validation_bsd100(BSD100_dir_LR_x2),
    #"BSD500_val": validation_bsd500_test(BSD500_test_dir),
    #"Urban100_HR": validation_urban100(Urban100_dir_HR)
}

# Noise adder
noise_adder = AugmentNoise(style=opt.noisetype)
# Masker
masker = Masker(width=4, mode='interpolate', mask_type='all')
# Network
network = UNet(in_channels=opt.n_channel,
                out_channels=opt.n_channel,
                wf=opt.n_feature)
if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()
# load pre-trained model
network = load_network(opt.checkpoint, network, strict=True)
beta = opt.beta

# turn on eval mode
network.eval()
# validation
save_test_path = os.path.join(opt.save_test_path, opt.log_name)
validation_path = os.path.join(save_test_path, "validation")
os.makedirs(validation_path, exist_ok=True)
np.random.seed(101)
# valid_repeat_times = {"Kodak24": 1, "BSD300": 1, "Set14": 1}
valid_repeat_times = {"Kodak24": 10, "BSD300": 3, "Set14": 20, "Urban100_LR_x4": 3,"BSD100_LR_x2":3, "BSD500_val":3}  # Add this line

for valid_name, valid_images in valid_dict.items():
    torch.cuda.empty_cache()
    save_dir = os.path.join(validation_path, valid_name)
    os.makedirs(save_dir, exist_ok=True)
    logger.info('Processing {} dataset'.format(valid_name))
    avg_psnr_dn = []
    avg_ssim_dn = []
    avg_psnr_exp = []
    avg_ssim_exp = []
    avg_psnr_mid = []
    avg_ssim_mid = []
    repeat_times = valid_repeat_times[valid_name]
    for i in range(repeat_times):
        for idx, im in enumerate(valid_images):
            origin255 = im.copy()
            origin255 = origin255.astype(np.uint8)
            im = np.array(im, dtype=np.float32) / 255.0
            noisy_im = noise_adder.add_valid_noise(im)
            noisy255 = noisy_im.copy()
            noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                255).astype(np.uint8)
            # padding to square
            H = noisy_im.shape[0]
            W = noisy_im.shape[1]
            val_size = (max(H, W) + 31) // 32 * 32
            noisy_im = np.pad(
                noisy_im,
                [[0, val_size - H], [0, val_size - W], [0, 0]],
                'reflect')
            transformer = transforms.Compose([transforms.ToTensor()])
            noisy_im = transformer(noisy_im)
            noisy_im = torch.unsqueeze(noisy_im, 0)
            noisy_im = noisy_im.cuda()
            with torch.no_grad():
                n, c, h, w = noisy_im.shape
                net_input, mask = masker.train(noisy_im)
                noisy_output = (network(net_input)*mask).view(n,-1,c,h,w).sum(dim=1)
                exp_output = network(noisy_im)
            pred_dn = noisy_output[:, :, :H, :W]
            pred_exp = exp_output[:, :, :H, :W]
            pred_mid = (pred_dn + beta*pred_exp) / (1 + beta)

            pred_dn = pred_dn.permute(0, 2, 3, 1)
            pred_exp = pred_exp.permute(0, 2, 3, 1)
            pred_mid = pred_mid.permute(0, 2, 3, 1)

            pred_dn = pred_dn.cpu().data.clamp(0, 1).numpy().squeeze(0)
            pred_exp = pred_exp.cpu().data.clamp(0, 1).numpy().squeeze(0)
            pred_mid = pred_mid.cpu().data.clamp(0, 1).numpy().squeeze(0)

            pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0,
                                255).astype(np.uint8)
            pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0,
                                255).astype(np.uint8)
            pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0,
                                255).astype(np.uint8)                   

            # calculate psnr
            psnr_dn = calculate_psnr(origin255.astype(np.float32),
                                        pred255_dn.astype(np.float32))
            avg_psnr_dn.append(psnr_dn)
            ssim_dn = calculate_ssim(origin255.astype(np.float32),
                                        pred255_dn.astype(np.float32))
            avg_ssim_dn.append(ssim_dn)

            psnr_exp = calculate_psnr(origin255.astype(np.float32),
                                        pred255_exp.astype(np.float32))
            avg_psnr_exp.append(psnr_exp)
            ssim_exp = calculate_ssim(origin255.astype(np.float32),
                                        pred255_exp.astype(np.float32))
            avg_ssim_exp.append(ssim_exp)

            psnr_mid = calculate_psnr(origin255.astype(np.float32),
                                        pred255_mid.astype(np.float32))
            avg_psnr_mid.append(psnr_mid)
            ssim_mid = calculate_ssim(origin255.astype(np.float32),
                                        pred255_mid.astype(np.float32))
            avg_ssim_mid.append(ssim_mid)

            logger.info(
                "{} - img:{}_{:03d} - PSNR_DN: {:.6f} dB; SSIM_DN: {:.6f}; PSNR_EXP: {:.6f} dB; SSIM_EXP: {:.6f}; PSNR_MID: {:.6f} dB; SSIM_MID: {:.6f}.".format(
                valid_name, i, idx, psnr_dn, ssim_dn, psnr_exp, ssim_exp, psnr_mid, ssim_mid
                )
            )

            # visualization
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}_clean.png".format(
                    i, idx))
            Image.fromarray(origin255).convert('RGB').save(
                save_path)
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}_noisy.png".format(
                    i, idx))
            Image.fromarray(noisy255).convert('RGB').save(
                save_path)
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}_dn.png".format(
                    i, idx))
            Image.fromarray(pred255_dn).convert('RGB').save(save_path)
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}_exp.png".format(
                    i, idx))
            Image.fromarray(pred255_exp).convert('RGB').save(save_path)
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}_mid.png".format(
                    i, idx))
            Image.fromarray(pred255_mid).convert('RGB').save(save_path)

    avg_psnr_dn = np.array(avg_psnr_dn)
    avg_psnr_dn = np.mean(avg_psnr_dn)
    avg_ssim_dn = np.mean(avg_ssim_dn)

    avg_psnr_exp = np.array(avg_psnr_exp)
    avg_psnr_exp = np.mean(avg_psnr_exp)
    avg_ssim_exp = np.mean(avg_ssim_exp)

    avg_psnr_mid = np.array(avg_psnr_mid)
    avg_psnr_mid = np.mean(avg_psnr_mid)
    avg_ssim_mid = np.mean(avg_ssim_mid)
    
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR_DN: {:.6f} dB; SSIM_DN: {:.6f}\n----PSNR_EXP: {:.6f} dB; SSIM_EXP: {:.6f}\n----PSNR_MID: {:.6f} dB; SSIM_MID: {:.6f}".format(
            valid_name, avg_psnr_dn, avg_ssim_dn, avg_psnr_exp, avg_ssim_exp, avg_psnr_mid, avg_ssim_mid
        )
    )