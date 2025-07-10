"""
FlowMo API Service
FastAPI backend for FlowMo image reconstruction with noise addition and metric calculation.
"""

import torch
import numpy as np
import base64
import io
import time
import logging
import os
import sys
from typing import Optional, Dict, Any
from PIL import Image
from torchvision import transforms

# FastAPI imports
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# FlowMo imports
import sys
flowmo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flowmo')
if flowmo_path not in sys.path:
    sys.path.insert(0, flowmo_path)
    
import train_utils
from omegaconf import OmegaConf
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and LPIPS
model = None
lpips_alex = None
lpips_vgg = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global model, lpips_alex, lpips_vgg
    
    # Startup
    try:
        logger.info("Initializing FlowMo API Service...")
        
        # Load FlowMo model
        model = load_flowmo_model()
        
        # Initialize LPIPS models
        lpips_alex, lpips_vgg = initialize_lpips_models()
        
        logger.info("FlowMo API Service initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FlowMo API Service...")

# FastAPI app
app = FastAPI(title="FlowMo API Service", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ImageProcessRequest(BaseModel):
    image: str  # base64 encoded image
    noise_level: float
    image_format: Optional[str] = "jpeg"

class MetricsData(BaseModel):
    psnr: float
    ssim: float
    lpips_alex: float
    lpips_vgg: float

class ImageProcessResponse(BaseModel):
    original_image: str
    noisy_image: str
    reconstructed_image: str
    metrics: Dict[str, MetricsData]
    processing_time: float
    noise_level: float

def load_flowmo_model(model_path: str = "flowmo_lo.pth", context_dim: int = 18):
    """Load a pretrained FlowMo model using the base config"""
    try:
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
        state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        
        # Use appropriate model state dict key
        model_key = 'model_ema_state_dict' if 'model_ema_state_dict' in state_dict else 'model_state_dict'
        model.load_state_dict(state_dict[model_key])
        
        # Move to GPU and set to eval mode
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        logger.info(f"FlowMo model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load FlowMo model: {str(e)}")
        raise

def initialize_lpips_models():
    """Initialize LPIPS models for metric calculation"""
    try:
        lpips_alex = lpips.LPIPS(net='alex')
        lpips_vgg = lpips.LPIPS(net='vgg')
        
        if torch.cuda.is_available():
            lpips_alex = lpips_alex.cuda()
            lpips_vgg = lpips_vgg.cuda()
        
        logger.info("LPIPS models initialized successfully")
        return lpips_alex, lpips_vgg
    except Exception as e:
        logger.error(f"Failed to initialize LPIPS models: {str(e)}")
        raise

def apply_awgn_to_image(image_tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Apply Additive White Gaussian Noise (AWGN) directly to image tensor.
    
    Args:
        image_tensor: Input image tensor in range [-1, 1], shape [B, C, H, W]
        noise_level: PSNR value (higher = less noise)
    
    Returns:
        Noisy image tensor on the same device as input
    """
    # Handle edge cases
    if noise_level >= 100:
        return image_tensor.clone()
    
    # Calculate signal power (average power per element)
    signal_power = torch.mean(image_tensor ** 2).item()
    
    # Convert PSNR to noise variance
    # PSNR = 10 * log10(P / σ²) where P is signal power
    # σ² = P / 10^(PSNR/10)
    psnr_linear = 10 ** (noise_level / 10)
    noise_variance = signal_power / psnr_linear
    noise_std = np.sqrt(noise_variance)
    
    # Generate noise with same shape as image
    noise = torch.randn_like(image_tensor) * noise_std
    
    # Add noise to the image
    noisy_image = image_tensor + noise
    
    # Clamp to maintain valid range
    noisy_image = torch.clamp(noisy_image, -1, 1)
    
    return noisy_image

def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Convert tensor to base64 string for frontend transmission"""
    # Convert from [-1, 1] to [0, 1] range
    tensor_01 = (tensor + 1.0) / 2.0
    tensor_01 = torch.clamp(tensor_01, 0, 1)
    
    # Convert to PIL Image
    # Assume tensor is [B, C, H, W] and we want the first image
    if tensor.dim() == 4:
        tensor_01 = tensor_01[0]  # Take first image from batch
    
    # Convert to numpy and transpose to HWC format
    img_np = tensor_01.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(img_np)
    
    # Convert to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64

def base64_to_tensor(base64_string: str) -> torch.Tensor:
    """Convert base64 string to tensor"""
    # Decode base64
    img_data = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    # Convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to expected input size
        transforms.ToTensor(),  # Converts to [0, 1]
    ])
    
    img_tensor = transform(pil_image)
    # Convert from [0, 1] to [-1, 1] range
    img_tensor = img_tensor * 2.0 - 1.0
    
    return img_tensor.unsqueeze(0)  # Add batch dimension

def calculate_metrics_eval(img1_tensor_01: torch.Tensor, img2_tensor_01: torch.Tensor, 
                          lpips_alex_model, lpips_vgg_model) -> Dict[str, float]:
    """
    Calculate LPIPS, PSNR, SSIM between two image tensors (range [0, 1]).
    """
    # LPIPS expects images in range [-1, 1]
    img1_lpips = img1_tensor_01 * 2.0 - 1.0
    img2_lpips = img2_tensor_01 * 2.0 - 1.0

    if torch.cuda.is_available():
        img1_lpips = img1_lpips.cuda()
        img2_lpips = img2_lpips.cuda()

    with torch.no_grad():
        lpips_alex_score = lpips_alex_model(img1_lpips, img2_lpips).mean().item()
        lpips_vgg_score = lpips_vgg_model(img1_lpips, img2_lpips).mean().item()

    psnr_scores = []
    ssim_scores = []
    
    for i in range(img1_tensor_01.shape[0]):
        img1_np = img1_tensor_01[i].permute(1, 2, 0).cpu().numpy()
        img2_np = img2_tensor_01[i].permute(1, 2, 0).cpu().numpy()

        img1_np = img1_np.astype(np.float32)
        img2_np = img2_np.astype(np.float32)

        psnr_val = psnr_loss(img1_np, img2_np, data_range=1.0)
        
        # For SSIM, ensure win_size is appropriate
        win_size = min(7, img1_np.shape[0] // 2, img1_np.shape[1] // 2)
        if win_size < 2:
            win_size = 2
        if win_size % 2 == 0:
            win_size -= 1
        if win_size <= 1:
            ssim_val = np.nan
        else:
            ssim_val = ssim_loss(img1_np, img2_np, data_range=1.0, channel_axis=-1, win_size=win_size)
        
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

    return {
        "lpips_alex": lpips_alex_score,
        "lpips_vgg": lpips_vgg_score,
        "psnr": np.nanmean(psnr_scores),
        "ssim": np.nanmean(ssim_scores),
    }

def run_complete_inference(model, image_tensor: torch.Tensor, noise_level: float, 
                         lpips_alex_model, lpips_vgg_model) -> Dict[str, Any]:
    """
    Complete inference pipeline with noise addition and reconstruction.
    
    Returns:
        Dict containing original, noisy, reconstructed images and metrics
    """
    with torch.no_grad():
        # Move image to appropriate device
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # 1. Create noisy image
        noisy_image = apply_awgn_to_image(image_tensor, noise_level)
        
        # 2. Run model reconstruction on original image
        dtype = torch.bfloat16 if train_utils.bfloat16_is_available() else torch.float32
        reconstructed = model.reconstruct_noise(image_tensor, noise_level, dtype=dtype)
        reconstructed = torch.clamp(reconstructed, -1, 1)
        
        # 3. Convert all images to [0, 1] range for metrics calculation
        original_01 = (image_tensor + 1.0) / 2.0
        noisy_01 = (noisy_image + 1.0) / 2.0
        reconstructed_01 = (reconstructed + 1.0) / 2.0
        
        # 4. Calculate metrics
        original_vs_noisy_metrics = calculate_metrics_eval(
            original_01, noisy_01, lpips_alex_model, lpips_vgg_model
        )
        
        original_vs_reconstructed_metrics = calculate_metrics_eval(
            original_01, reconstructed_01, lpips_alex_model, lpips_vgg_model
        )
        
        return {
            "original": image_tensor,
            "noisy": noisy_image,
            "reconstructed": reconstructed,
            "metrics": {
                "original_vs_noisy": original_vs_noisy_metrics,
                "original_vs_reconstructed": original_vs_reconstructed_metrics
            }
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "lpips_loaded": lpips_alex is not None and lpips_vgg is not None
    }

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "FlowMo",
        "context_dim": 18,
        "patch_size": 4,
        "input_size": 256,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/process_image", response_model=ImageProcessResponse)
async def process_image(request: ImageProcessRequest):
    """Main image processing endpoint"""
    if model is None or lpips_alex is None or lpips_vgg is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = time.time()
    
    try:
        # Convert base64 to tensor
        image_tensor = base64_to_tensor(request.image)
        
        # Run complete inference
        results = run_complete_inference(
            model, image_tensor, request.noise_level, lpips_alex, lpips_vgg
        )
        
        # Convert tensors to base64 for response
        original_b64 = tensor_to_base64(results["original"])
        noisy_b64 = tensor_to_base64(results["noisy"])
        reconstructed_b64 = tensor_to_base64(results["reconstructed"])
        
        processing_time = time.time() - start_time
        
        # Format metrics response
        metrics_response = {
            "original_vs_noisy": MetricsData(**results["metrics"]["original_vs_noisy"]),
            "original_vs_reconstructed": MetricsData(**results["metrics"]["original_vs_reconstructed"])
        }
        
        return ImageProcessResponse(
            original_image=original_b64,
            noisy_image=noisy_b64,
            reconstructed_image=reconstructed_b64,
            metrics=metrics_response,
            processing_time=processing_time,
            noise_level=request.noise_level
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4040)
