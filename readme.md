# SemanticComm - Image Reconstruction API & Frontend

A complete web-based interface for SemCom  (Flow Matching) image reconstruction with noise addition and comprehensive metric evaluation. This project provides both a FastAPI backend service and a responsive web frontend for semantic communication research.

## 🎯 Overview

SemanticComm implements a semantic communication system using MMDit (Flow Matching) models for image reconstruction. The system can:

- Add AWGN (Additive White Gaussian Noise) to images
- Reconstruct images using pre-trained FlowMo models
- Calculate comprehensive image quality metrics (PSNR, SSIM, LPIPS)
- Provide a web interface for easy experimentation
- Serve as an API backend for integration with other systems

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FlowMo model checkpoint (`flowmo_lo.pth`)
- Base configuration file (`flowmo/configs/base.yaml`)

### Installation

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd semacomm
   ```

2. **Install dependencies**:
   ```bash
   # Install API service dependencies
   pip install -r requirements_api.txt
   
   # Install core FlowMo dependencies (if not already installed)
   pip install torch torchvision omegaconf lpips scikit-image
   ```

3. **Start the API service**:
   ```bash
   python api_service.py
   ```
   The API will be available at `http://localhost:8000`

4. **Start the frontend** (in a new terminal):
   ```bash
   cd frontend
   python -m http.server 8080
   ```
   The web interface will be available at `http://localhost:8080`

## 📁 Project Structure

```
semacomm/
├── api_service.py              # FastAPI backend service
├── requirements_api.txt        # API-specific dependencies
├── PROJECT_README.md           # Detailed project documentation
├── TODO.md                     # Implementation progress tracking
├── GEMINI.md                   # AI agent instructions
├── ui.md                       # UI implementation details
├── data_log.txt               # Sample evaluation results
│
├── flowmo/                     # FlowMo model implementation
│   ├── data.py                # Dataset handling
│   ├── train.py               # Training pipeline
│   ├── models.py              # Model definitions
│   ├── train_utils.py         # Training utilities
│   └── configs/
│       └── base.yaml          # Model configuration
│
├── frontend/                   # Web frontend
│   ├── index.html             # Main web interface
│   ├── README.md              # Frontend documentation
│   └── static/
│       ├── css/style.css      # Styling
│       └── js/main.js         # JavaScript functionality
│
├── OLD_x/                     # Legacy evaluation code
│   └── drone_eval.py          # Drone dataset evaluation
│
├── Eval_data/                 # Evaluation datasets
└── flowmo_lo.pth             # Pre-trained FlowMo model
```

## 🛠 Features

### Backend API Service

- **Model Management**: Automatic FlowMo model loading with CUDA support
- **Image Processing**: Base64 image conversion and preprocessing
- **Noise Simulation**: AWGN noise application for communication channel simulation
- **Metric Calculation**: Comprehensive quality metrics (PSNR, SSIM, LPIPS Alex/VGG)
- **RESTful API**: FastAPI with automatic documentation at `/docs`
- **CORS Support**: Cross-origin requests for frontend integration

### Web Frontend

- **Drag & Drop Upload**: Intuitive image upload with real-time preview
- **Noise Control**: Adjustable noise level slider (-5 to +20 PSNR)
- **Three-Panel Display**: Side-by-side comparison of original, noisy, and reconstructed images
- **Real-time Metrics**: Live display of image quality metrics
- **Size Information**: Transmission size calculations and compression ratios
- **Responsive Design**: Works across desktop and mobile devices

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Returns service status and model loading state.

### Model Information
```
GET /model_info
```
Returns details about the loaded FlowMo model.

### Image Processing
```
POST /process_image
```
Main endpoint for image reconstruction pipeline.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "noise_level": 10.0,
  "image_format": "jpeg"
}
```

**Response:**
```json
{
  "original_image": "base64_string",
  "noisy_image": "base64_string", 
  "reconstructed_image": "base64_string",
  "metrics": {
    "original_vs_noisy": {
      "psnr": 15.2,
      "ssim": 0.65,
      "lpips_alex": 0.12,
      "lpips_vgg": 0.15
    },
    "original_vs_reconstructed": {
      "psnr": 22.8,
      "ssim": 0.78,
      "lpips_alex": 0.08,
      "lpips_vgg": 0.11
    }
  },
  "processing_time": 0.85,
  "noise_level": 10.0
}
```

## 📊 Evaluation and Metrics

The system calculates multiple image quality metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better
- **SSIM (Structural Similarity Index)**: Range [0,1], higher is better  
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Lower is better
  - Alex: AlexNet-based perceptual metric
  - VGG: VGG-based perceptual metric

## 🧪 Usage Examples

### Python API Client
```python
import requests
import base64

# Load and encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Make API request
response = requests.post("http://localhost:8000/process_image", json={
    "image": image_b64,
    "noise_level": 15.0
})

result = response.json()
print(f"PSNR: {result['metrics']['original_vs_reconstructed']['psnr']:.2f}")
```

### Web Interface
1. Open `http://localhost:8080` in your browser
2. Drag and drop an image or click to upload
3. Adjust the noise level slider
4. Click "Process Image" to see results
5. View original, noisy, and reconstructed images with metrics

## 🔬 Research Applications

This system is designed for semantic communication research, including:

- **Channel Simulation**: AWGN noise modeling for wireless communication
- **Compression Analysis**: Transmission size vs. quality trade-offs
- **Model Evaluation**: Comparative analysis of different reconstruction models
- **Quality Assessment**: Multi-metric evaluation of reconstruction performance

## 📈 Performance

Sample evaluation results (from `data_log.txt`):
- LPIPS Alex: 0.140
- LPIPS VGG: 0.172  
- PSNR: 17.08 dB
- SSIM: 0.562
- Noise Level: 7 PSNR

## 🤝 Contributing

This project follows structured development practices:
- Check `TODO.md` for current implementation status
- Review `GEMINI.md` for AI agent guidelines
- See `ui.md` for frontend implementation details


## 🙏 Acknowledgments

Built on the MMDIT architecture and FlowMo training (Flow Matching) framework for generative modeling and semantic communication research.

---

For detailed API documentation, visit `http://localhost:8000/docs` when the service is running.
