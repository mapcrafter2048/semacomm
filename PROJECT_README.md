# FlowMo Image Reconstruction API & Frontend

This project provides a complete web-based interface for FlowMo image reconstruction, including noise addition and comprehensive metric evaluation. It consists of a FastAPI backend service and a responsive web frontend.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FlowMo model checkpoint (`flowmo_lo.pth`)
- Base configuration file (`flowmo/configs/base.yaml`)

### Installation

1. **Clone and Setup**:
   ```bash
   cd /home/utka/proj/semacomm
   
   # Install API dependencies
   pip install -r requirements_api.txt
   
   # Ensure existing FlowMo dependencies are installed
   # torch, torchvision, omegaconf, lpips, scikit-image, etc.
   ```

2. **Start the API Service**:
   ```bash
   python api_service.py
   ```
   The API will be available at `http://localhost:8000`

3. **Start the Frontend**:
   ```bash
   cd frontend
   python -m http.server 8080
   ```
   The frontend will be available at `http://localhost:8080`

## 📁 Project Structure

```
semacomm/
├── api_service.py              # FastAPI backend service
├── requirements_api.txt        # API dependencies
├── README.md                   # This file
├── TODO.md                     # Implementation checklist
│
├── flowmo/                     # FlowMo model code
│   ├── infer.py               # Original inference pipeline
│   ├── models.py              # Model definitions
│   ├── train_utils.py         # Training utilities
│   └── configs/
│       └── base.yaml          # Model configuration
│
├── frontend/                   # Web frontend
│   ├── index.html             # Main interface
│   ├── README.md              # Frontend documentation
│   └── static/
│       ├── css/style.css      # Styling
│       └── js/main.js         # JavaScript functionality
│
├── flowmo_lo.pth              # FlowMo model checkpoint
└── Eval_data/                 # Evaluation datasets
```

## 🛠 Features

### Backend API (`api_service.py`)

- **Model Management**: Automatic FlowMo model loading and CUDA handling
- **Image Processing**: Base64 image conversion and preprocessing
- **Noise Addition**: AWGN noise application directly to images
- **Comprehensive Metrics**: PSNR, SSIM, LPIPS (Alex & VGG) calculations
- **RESTful API**: FastAPI with automatic documentation
- **CORS Support**: Cross-origin requests for frontend integration

### Frontend Interface

- **Drag & Drop Upload**: Intuitive image upload with preview
- **Noise Control**: Adjustable noise level (-5 to +20 PSNR)
- **Three-Panel Results**: Original, noisy, and reconstructed images
- **Real-time Metrics**: Live display of image quality metrics
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Comprehensive error messages and recovery

## 🔌 API Endpoints

### `GET /health`
Check API service health and model status.

**Response:**
```json
{
  "status": "healthy",
  "cuda_available": true,
  "model_loaded": true,
  "lpips_loaded": true
}
```

### `GET /model_info`
Get information about the loaded FlowMo model.

**Response:**
```json
{
  "model_type": "FlowMo",
  "context_dim": 18,
  "patch_size": 4,
  "input_size": 256,
  "device": "cuda"
}
```

### `POST /process_image`
Main image processing endpoint.

**Request:**
```json
{
  "image": "base64_encoded_image_data",
  "noise_level": 7.0,
  "image_format": "jpeg"
}
```

**Response:**
```json
{
  "original_image": "base64_encoded_original",
  "noisy_image": "base64_encoded_noisy", 
  "reconstructed_image": "base64_encoded_reconstructed",
  "metrics": {
    "original_vs_noisy": {
      "psnr": 15.23,
      "ssim": 0.45,
      "lpips_alex": 0.32,
      "lpips_vgg": 0.28
    },
    "original_vs_reconstructed": {
      "psnr": 22.45,
      "ssim": 0.78,
      "lpips_alex": 0.18,
      "lpips_vgg": 0.15
    }
  },
  "processing_time": 2.34,
  "noise_level": 7.0
}
```

## 📊 Metrics Explained

- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **LPIPS**: Learned Perceptual Image Patch Similarity (lower is better)
  - Alex: Using AlexNet features
  - VGG: Using VGG features

## 🧪 Usage Example

1. **Upload an image** through the web interface
2. **Adjust noise level** using the slider (e.g., 7 for moderate noise)
3. **Click "Process Image"** to start reconstruction
4. **View results**:
   - Original image as uploaded
   - Noisy image with applied AWGN
   - Reconstructed image from FlowMo
   - Metrics comparing image quality

## 🔧 Configuration

### Model Configuration

Edit `flowmo/configs/base.yaml` to adjust model parameters:

```yaml
model:
  context_dim: 18        # Context dimension (18 for lo, 56 for hi)
  patch_size: 4          # Patch size for processing
  code_length: 256       # Code length for quantization
```

### API Configuration

Modify `api_service.py` for custom settings:

```python
# Model loading
model_path = "flowmo_lo.pth"  # Path to checkpoint
context_dim = 18              # Context dimension

# Server settings
host = "0.0.0.0"             # Bind address
port = 8000                  # Port number
```

## 🐛 Troubleshooting

### Common Issues

1. **"Model not loaded" error**:
   - Ensure `flowmo_lo.pth` is in the correct location
   - Check `flowmo/configs/base.yaml` exists
   - Verify CUDA availability if using GPU

2. **"Cannot connect to API" error**:
   - Ensure API service is running on `http://localhost:8000`
   - Check firewall settings
   - Verify CORS configuration

3. **Image processing fails**:
   - Check image format (JPEG, PNG, WEBP supported)
   - Ensure image size is under 10MB
   - Verify GPU memory availability

4. **Slow processing**:
   - Use CUDA if available
   - Reduce image size
   - Check system resources

### Development

For development and testing:

```bash
# Install development dependencies
pip install httpx pytest

# Test API endpoints
python -c "
import httpx
response = httpx.get('http://localhost:8000/health')
print(response.json())
"

# View API documentation
# Open http://localhost:8000/docs in browser
```

## 📝 License

This project extends the FlowMo codebase for web-based image reconstruction. Please refer to the original FlowMo license for usage terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review API documentation at `http://localhost:8000/docs`
3. Examine browser console for frontend errors
4. Check API service logs for backend issues
