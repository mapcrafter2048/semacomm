import argparse
import os
import glob
import random
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf

import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from flowmo import train_utils

# --- Helper Functions ---

def process_image_for_eval(image_path, target_size=256):
    """Load and preprocess an image for FlowMo evaluation."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])
    img_tensor = transform(image)
    # Convert from [0, 1] to [-1, 1] range
    img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor.unsqueeze(0)  # Add batch dimension




def add_awgn(signal, snr_db=10):
    """
    Adds AWGN to a signal.

    Args:
        signal (numpy.ndarray): The input signal.
        snr_db (float): The desired signal-to-noise ratio in dB.

    Returns:
        numpy.ndarray: The noisy signal.
    """
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(signal))
    noisy_signal = signal + noise
    return noisy_signal

def simulate_udp_packet_loss_eval(image_tensor_cuda, loss_rate=0.05, patch_size=32):
    """
    Simulates packet loss on a CUDA image tensor by zeroing random patches.
    Args:
        image_tensor_cuda (torch.Tensor): CUDA tensor of shape [B, C, H, W].
        loss_rate (float): Fraction of the image area to corrupt (0.0 to 1.0).
        patch_size (int): Size of square patch to drop.
    Returns:
        torch.Tensor: A new tensor with simulated packet loss.
    """
    assert image_tensor_cuda.device.type == 'cuda', "Image must be on CUDA"
    B, C, H, W = image_tensor_cuda.shape
    corrupted_image = image_tensor_cuda.clone()

    num_patches_per_image = int((H * W * loss_rate) / (patch_size ** 2))

    for i in range(B):
        for _ in range(num_patches_per_image):
            # Ensure patch does not go out of bounds
            if H - patch_size < 0 or W - patch_size < 0:
                # This can happen if patch_size is larger than image dimension
                # Or if H or W is 0, though less likely for typical images
                print(f"Warning: Patch size {patch_size} may be too large for image dimensions {H}x{W}. Skipping patch.")
                continue
            y = random.randint(0, H - patch_size)
            x = random.randint(0, W - patch_size)
            corrupted_image[i, :, y:y+patch_size, x:x+patch_size] = 0  # Zero out patch
    return corrupted_image

def calculate_metrics_eval(img1_tensor_01, img2_tensor_01, lpips_alex, lpips_vgg):
    """
    Calculate LPIPS, PSNR, SSIM between two image tensors (range [0, 1]).
    Args:
        img1_tensor_01 (torch.Tensor): First image tensor, shape [B, C, H, W], range [0, 1].
        img2_tensor_01 (torch.Tensor): Second image tensor, shape [B, C, H, W], range [0, 1].
        lpips_alex: Pre-initialized LPIPS Alex model.
        lpips_vgg: Pre-initialized LPIPS VGG model.
    Returns:
        dict: Dictionary with metric scores.
    """
    # LPIPS expects images in range [-1, 1]
    img1_lpips = img1_tensor_01 * 2.0 - 1.0
    img2_lpips = img2_tensor_01 * 2.0 - 1.0

    if torch.cuda.is_available():
        img1_lpips = img1_lpips.cuda()
        img2_lpips = img2_lpips.cuda()

    with torch.no_grad():
        lpips_alex_score = lpips_alex(img1_lpips, img2_lpips).mean().item()
        lpips_vgg_score = lpips_vgg(img1_lpips, img2_lpips).mean().item()

    psnr_scores = []
    ssim_scores = []
    
    for i in range(img1_tensor_01.shape[0]):
        img1_np = img1_tensor_01[i].permute(1, 2, 0).cpu().numpy()
        img2_np = img2_tensor_01[i].permute(1, 2, 0).cpu().numpy()

        img1_np = img1_np.astype(np.float32)
        img2_np = img2_np.astype(np.float32)

        psnr_val = psnr_loss(img1_np, img2_np, data_range=1.0)
        # For SSIM, ensure win_size is appropriate and less than image dimensions
        win_size = min(7, img1_np.shape[0] // 2, img1_np.shape[1] // 2) # Ensure win_size is odd and < image dim
        if win_size < 2 : win_size = 2 # if image is too small, use a minimal win_size
        if win_size % 2 == 0: win_size -=1 # ensure odd
        if win_size <=1: # if win_size is too small, ssim might error or be meaningless
             ssim_val = np.nan # or some other placeholder like 0 or 1 depending on how you want to treat it
        else:
            ssim_val = ssim_loss(img1_np, img2_np, data_range=1.0, channel_axis=-1, win_size=win_size)
        
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

    return {
        "lpips_alex": lpips_alex_score,
        "lpips_vgg": lpips_vgg_score,
        "psnr": np.nanmean(psnr_scores), # Use nanmean in case of nan ssim scores
        "ssim": np.nanmean(ssim_scores), # Use nanmean
    }

# --- Main Evaluation Logic ---
def main():
    parser = argparse.ArgumentParser(description="FlowMo Drone Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FlowMo checkpoint")
    parser.add_argument("--context_dim", type=int, default=18, help="Context dimension for FlowMo model")
    parser.add_argument("--data_dir", type=str, default="Eval_data", help="Directory containing evaluation images")
    parser.add_argument("--output_dir", type=str, default="drone_eval_output", help="Directory to save results and plots")
    parser.add_argument("--image_size", type=int, default=256, help="Target size for images")
    parser.add_argument("--config_path", type=str, default="flowmo/configs/base.yaml", help="Path to base model config")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    lpips_alex = lpips.LPIPS(net='alex')
    lpips_vgg = lpips.LPIPS(net='vgg')
    if torch.cuda.is_available():
        lpips_alex = lpips_alex.cuda()
        lpips_vgg = lpips_vgg.cuda()

    print(f"Loading model from {args.checkpoint}...")

    loss_rates = [0.05, 0.1, 0.2, 0.4]
    patch_sizes = [8, 16, 32, 64]
    udp_conditions = []
    for lr in loss_rates:
        for ps in patch_sizes:
            udp_conditions.append({"loss_rate": lr, "patch_size": ps, "name": f"udp_lr{lr}_ps{ps}"})

    all_results = []

    image_paths = glob.glob(os.path.join(args.data_dir, "*.jpg")) + \
                  glob.glob(os.path.join(args.data_dir, "*.png")) + \
                  glob.glob(os.path.join(args.data_dir, "*.jpeg"))

    print(f"Found {len(image_paths)} images in {args.data_dir}")

    for image_path in tqdm(image_paths, desc="Processing images"):
        image_name = os.path.basename(image_path)
        try:
            original_image_tensor_minus1_1 = process_image_for_eval(image_path, args.image_size)
        except Exception as e:
            print(f"Skipping image {image_name} due to processing error: {e}")
            continue
        
        if torch.cuda.is_available():
            original_image_tensor_minus1_1 = original_image_tensor_minus1_1.cuda()

        original_image_tensor_01 = (original_image_tensor_minus1_1 + 1.0) / 2.0

        with torch.no_grad():
            reconstructed_image_tensor_minus1_1 = model.reconstruct(original_image_tensor_minus1_1, dtype=dtype)
            reconstructed_image_tensor_minus1_1 = reconstructed_image_tensor_minus1_1.clamp(-1, 1)
        
        reconstructed_image_tensor_01 = (reconstructed_image_tensor_minus1_1 + 1.0) / 2.0
        
        metrics_recon = calculate_metrics_eval(original_image_tensor_01, reconstructed_image_tensor_01, lpips_alex, lpips_vgg)
        all_results.append({
            "image_name": image_name,
            "condition_name": "model_reconstruction",
            **metrics_recon
        })

        for condition in udp_conditions:
            corrupted_image_udp_tensor_minus1_1 = simulate_udp_packet_loss_eval(
                original_image_tensor_minus1_1, 
                loss_rate=condition["loss_rate"], 
                patch_size=condition["patch_size"]
            )
            corrupted_image_udp_tensor_01 = (corrupted_image_udp_tensor_minus1_1 + 1.0) / 2.0

            metrics_udp = calculate_metrics_eval(original_image_tensor_01, corrupted_image_udp_tensor_01, lpips_alex, lpips_vgg)
            all_results.append({
                "image_name": image_name,
                "condition_name": condition["name"],
                **metrics_udp
            })
    
    if not all_results:
        print("No results to process. Exiting.")
        return

    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Select only numeric columns for aggregation
    numeric_metric_columns = ["lpips_alex", "lpips_vgg", "psnr", "ssim"]
    # Ensure all expected numeric columns are present before trying to aggregate
    valid_numeric_columns = [col for col in numeric_metric_columns if col in results_df.columns]
    
    if not valid_numeric_columns:
        print("Error: No valid numeric metric columns found for aggregation. Exiting plotting.")
        return

    aggregated_results = results_df.groupby("condition_name")[valid_numeric_columns].mean().reset_index()
    
    condition_order = ["model_reconstruction"] + [c["name"] for c in udp_conditions]

    metrics_to_plot = ["lpips_alex", "lpips_vgg", "psnr", "ssim"]
    for metric in metrics_to_plot:
        if metric not in aggregated_results.columns:
            print(f"Metric {metric} not found in aggregated results. Skipping plot.")
            continue
        plt.figure(figsize=(20, 8))
        bars = plt.bar(aggregated_results["condition_name"], aggregated_results[metric])
        
        for bar in bars:
            yval = bar.get_height()
            if pd.notna(yval):
                 plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >=0 else 'top', ha='center', fontsize=8)

        plt.xlabel("Condition")
        plt.ylabel(metric.upper())
        plt.title(f"Average {metric.upper()} Comparison (Lower LPIPS is better, Higher PSNR/SSIM is better)")
        plt.xticks(rotation=90, ha="right")
        plt.subplots_adjust(bottom=0.3)
        plt.grid(axis='y', linestyle='--')
        plot_path = os.path.join(args.output_dir, f"{metric}_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

    print("Evaluation complete.")

if __name__ == "__main__":
    main()

