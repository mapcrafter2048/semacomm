import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import random
import train_utils
from omegaconf import OmegaConf
import logging
import json
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss



log_state_dict = False  # Set to True to log state_dict inspection
log_prinout = True  # Set to True to log print statements
def load_flowmo_model(model_path, context_dim=18):
    """Load a pretrained FlowMo model using the base config"""
    base_config = OmegaConf.load("flowmo/configs/base.yaml")
    
    # Override only what's needed
    override_config = OmegaConf.create({
        "model": {
            "context_dim": context_dim,
            "enable_mup": False  # Disable MUP for inference
        }
    })
    
    # Merge the configs
    config = OmegaConf.merge(base_config, override_config)
    # Build model
    model = train_utils.build_model(config)
    
    # Load weights
    # Set up logging to save state_dict inspection to a file
    if log_state_dict:
        logging.basicConfig(filename="state_dict_inspection.log", level=logging.INFO, format="%(message)s")

    # Load the state_dict
    state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

    # Inspect and log the keys and structure of state_dict
    if log_state_dict:
        logging.info("State Dict Keys:")
        logging.info(json.dumps(list(state_dict.keys()), indent=4))

        # Optionally, log the structure of a specific key (e.g., 'model_state_dict')
    
        if 'model_state_dict' in state_dict:
            logging.info("Structure of 'model_state_dict':")
            logging.info(json.dumps({k: v.shape if hasattr(v, 'shape') else str(type(v)) for k, v in state_dict['model_state_dict'].items()}, indent=4))
    
    model_key = 'model_ema_state_dict' if 'model_ema_state_dict' in state_dict else 'model_state_dict'
    model.load_state_dict(state_dict[model_key])
    
    model = model.cuda().eval()
    return model


def simulate_udp_packet_loss(image, loss_rate=0.05, patch_size=32):
    """
    Simulates packet loss on a CUDA image tensor by zeroing random patches.

    Args:
        image (torch.Tensor): CUDA tensor of shape [C, H, W].
        loss_rate (float): Fraction of the image area to corrupt (0.0 to 1.0).
        patch_size (int): Size of square patch to drop.

    Returns:
        torch.Tensor: A new tensor with simulated packet loss.
    """
    assert image.device.type == 'cuda', "Image must be on CUDA"
    C, H, W = image.shape
    corrupted_image = image.clone()

    num_patches = int((H * W * loss_rate) / (patch_size ** 2))

    for _ in range(num_patches):
        y = random.randint(0, H - patch_size)
        x = random.randint(0, W - patch_size)
        corrupted_image[:, y:y+patch_size, x:x+patch_size] = 0  # Zero out patch (simulate loss)

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





def process_image(image_path, target_size=256):
    """Load and preprocess an image for FlowMo"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),  # Converts to [0, 1]
    ])
    
    img_tensor = transform(image)
    # Convert from [0, 1] to [-1, 1] range
    img_tensor = img_tensor * 2.0 - 1.0
    
    return img_tensor.unsqueeze(0)  # Add batch dimension


def run_inference(model, image, lpips_alex, lpips_vgg, Noise_level):
    """Run FlowMo inference and compute MSE loss"""
    with torch.no_grad():
        # Move image to GPU with correct format
        image_cuda = image.cuda()
        
        # Get optimal dtype for inference
        dtype = torch.bfloat16 if train_utils.bfloat16_is_available() else torch.float32
        

        # Reconstruct image
        reconstructed = model.reconstruct_noise(image_cuda, Noise_level, dtype=dtype)
        # reconstructed = model.reconstruct(image_cuda, dtype=dtype)
        # reconstructed = reconstructed.clamp(-1, 1)
        
        #calculte metrics 
        metrics = calculate_metrics_eval(
            img1_tensor_01=image_cuda,
            img2_tensor_01=reconstructed,
            lpips_alex=lpips_alex,
            lpips_vgg=lpips_vgg
        )
        # Log metrics to file
        
        # Configure logging if not already done
        logging.basicConfig(
            filename='data_log.txt',
            level=logging.INFO,
            format='%(message)s',
            filemode='a'  # Append mode to add new lines
        )
        
        # Log the metrics
        log_message = f"LPIPS Alex: {metrics['lpips_alex']:.6f}, LPIPS VGG: {metrics['lpips_vgg']:.6f}, PSNR: {metrics['psnr']:.6f}, SSIM: {metrics['ssim']:.6f}, Noise Level: {Noise_level}"
        logging.info(log_message)
        
        # Also print to console if needed
        print(log_message)

        """      
        # Calculate MSE loss (in normalized space)
        orig_recons_mse = F.mse_loss(image_cuda, reconstructed).item()
        orig_udp_mse =  F.mse_loss(image_cuda, corrupt_image_cuda).item()
        recons_udp_mse = F.mse_loss(reconstructed, corrupt_image_cuda).item()
        


        # Convert back to [0, 1] range for visualization
        image_vis = (image_cuda + 1.0) / 2.0
        recon_vis = (reconstructed + 1.0) / 2.0

        return {
            'original': image_vis.cpu(),
            'reconstructed': recon_vis.cpu(),
            'mse_loss': orig_recons_mse,
            'udp_mse_loss': orig_udp_mse,
            'recons_udp_mse_loss': recons_udp_mse,
        }
        """
        metrics["original"] = ((image_cuda + 1.0) / 2.0)
        metrics["reconstructed"] = ((reconstructed + 1.0) / 2.0)

        return metrics


def save_comparison(result, output_path="flowmo_comparison.png"):
    """Save original and reconstructed image side by side"""
    from torchvision.utils import save_image

    
    # Create a grid of images: [original, reconstructed]
    grid = torch.cat([result['original'], result['reconstructed']], dim=0)
    save_image(grid, output_path, nrow=2)
    print(f"Comparison saved to {output_path}")



## MAIN FUNCTION WITH PARSER CODE
def main():
    parser = argparse.ArgumentParser(description="FlowMo Single Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FlowMo checkpoint")
    parser.add_argument("--context_dim", type=int, default=18, 
                        help="Context dimension (18 for flowmo_lo, 56 for flowmo_hi)")
    parser.add_argument("--output", type=str, default="flowmo_comparison.png", help="Output path")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_flowmo_model(args.checkpoint, args.context_dim)
    
    # Process image
    print(f"Processing image: {args.image}")
    image = process_image(args.image)
    
    lpips_alex = lpips.LPIPS(net='alex')
    lpips_vgg = lpips.LPIPS(net='vgg')
    if torch.cuda.is_available():
        lpips_alex = lpips_alex.cuda()
        lpips_vgg = lpips_vgg.cuda()


    # Run inference
    print("Running inference...")
    result = run_inference(model, image, lpips_alex, lpips_vgg)
        
    # Save comparison
    save_comparison(result, args.output)


def main_simple():
    """Simple main function with default values, no argument parsing"""
    # Default values
    # checkpoint_path = "flowmo_lo.pth"
    checkpoint_path = "flowmo_lo.pth"
    context_dim = 18
    # image_path = "Eval_data/VisDrone2019-MOT-train_uav0000013_01392_v_0000001.jpg"  # Default image path
    image_path = "/home/utka/Downloads/n01514859_hen.JPEG"
    output_path = "Image_net.png"  # Default output path
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_flowmo_model(checkpoint_path, context_dim)
    
    # Process image
    print(f"Processing image: {image_path}")
    image = process_image(image_path)
    
    lpips_alex = lpips.LPIPS(net='alex')
    lpips_vgg = lpips.LPIPS(net='vgg')
    if torch.cuda.is_available():
        lpips_alex = lpips_alex.cuda()
        lpips_vgg = lpips_vgg.cuda()

    # Run inference
    # for i in [-5,7,2]:
    #     print("Running inference...")
    #     result = run_inference(model, image, lpips_alex, lpips_vgg, Noise_level=i)

    result = run_inference(model, image, lpips_alex, lpips_vgg, Noise_level=7)

    # Save comparison
    save_comparison(result, output_path)
    


if __name__ == "__main__":
    main_simple()