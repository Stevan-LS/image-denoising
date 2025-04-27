import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_compare_images(benchmark_dir, repeat_index = 0, image_idx = 0):
    # File patterns
    noisy_pattern = f"{repeat_index:03d}-{image_idx:03d}_noisy.png"
    denoised_pattern = f"{repeat_index:03d}-{image_idx:03d}_mid.png"
    clean_pattern = f"{repeat_index:03d}-{image_idx:03d}_clean.png"
    
    # Load images
    clean_path = os.path.join(benchmark_dir, clean_pattern)
    noisy_path = os.path.join(benchmark_dir, noisy_pattern)
    denoised_path = os.path.join(benchmark_dir, denoised_pattern)
    
    clean_img = np.array(Image.open(clean_path))
    noisy_img = np.array(Image.open(noisy_path))
    denoised_img = np.array(Image.open(denoised_path))
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    
    ax1.imshow(clean_img, cmap='gray')
    ax1.set_title('Clean Image')
    ax1.axis('off')

    ax2.imshow(noisy_img, cmap='gray')
    ax2.set_title('Noisy Image')
    ax2.axis('off')
    
    ax3.imshow(denoised_img, cmap='gray')
    ax3.set_title('Denoised Image')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set the path to your SIDD_Benchmark directory
    benchmark_dir = "BSD100_bsd_forge"
    
    # Create full image comparison
    load_and_compare_images(benchmark_dir, repeat_index=0, image_idx=87)