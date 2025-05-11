import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from flowmo import train_utils
from omegaconf import OmegaConf


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
    state_dict = torch.load(model_path, map_location='cpu')
    model_key = 'model_ema_state_dict' if 'model_ema_state_dict' in state_dict else 'model_state_dict'
    model.load_state_dict(state_dict[model_key])
    
    model = model.cuda().eval()
    return model


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


def run_inference(model, image):
    """Run FlowMo inference and compute MSE loss"""
    with torch.no_grad():
        # Move image to GPU with correct format
        image_cuda = image.cuda()
        
        # Get optimal dtype for inference
        dtype = torch.bfloat16 if train_utils.bfloat16_is_available() else torch.float32
        
        # Reconstruct image
        reconstructed = model.reconstruct(image_cuda, dtype=dtype)
        reconstructed = reconstructed.clamp(-1, 1)
        
        # Calculate MSE loss (in normalized space)
        mse_loss = F.mse_loss(image_cuda, reconstructed).item()
        
        # Convert back to [0, 1] range for visualization
        image_vis = (image_cuda + 1.0) / 2.0
        recon_vis = (reconstructed + 1.0) / 2.0
        
        return {
            'original': image_vis.cpu(),
            'reconstructed': recon_vis.cpu(),
            'mse_loss': mse_loss
        }


def save_comparison(result, output_path="flowmo_comparison.png"):
    """Save original and reconstructed image side by side"""
    from torchvision.utils import save_image
    
    # Create a grid of images: [original, reconstructed]
    grid = torch.cat([result['original'], result['reconstructed']], dim=0)
    save_image(grid, output_path, nrow=2)
    print(f"Comparison saved to {output_path}")


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
    
    # Run inference
    print("Running inference...")
    result = run_inference(model, image)
    
    # Print results
    print(f"MSE Loss: {result['mse_loss']:.6f}")
    
    # Save comparison
    save_comparison(result, args.output)


if __name__ == "__main__":
    main()