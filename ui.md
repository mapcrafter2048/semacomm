OBJECTIVE: 
~~~
Using infer.py to understand inference pipeline, create another file called api_service.py
This file works as an api reciever for inputs and outputs. 
Use FastAPI to create api service.
When the api service is active, it loads the model in background and waits for the new image input and noise level. 
The image input is provided by frontend service. The image would be loaded in frontend and would be passed as input. 
Design the frontend and api to handle same inputs (ie same image format like tensor, or pillow image). 
So there is no compatibility issue
The API returns regenerated image, and scores across various metrics. 
Also in inference pipeline create another function where we apply AWGN noise level to orignal image and calculate the scroes between original and noisy image, then show that image along with scores in 
~~~


## **Detailed Implementation Plan for FlowMo API Service and Frontend**

### **Project Structure Analysis**

Based on the code analysis, the FlowMo inference pipeline works as follows:

1. **Model Loading**: Uses `load_flowmo_model` to load pretrained model with config
2. **Image Processing**: `process_image` converts PIL images to tensor format [-1, 1]
3. **Noise Addition**: `apply_awgn_noise` adds AWGN noise to latent codes
4. **Reconstruction**: `model.reconstruct_noise` generates reconstructed images
5. **Metrics Calculation**: `calculate_metrics_eval` computes LPIPS, PSNR, SSIM

### **Implementation Plan**

#### **Phase 1: Backend API Service (`api_service.py`)**

**Location**: `/home/utka/proj/semacomm/api_service.py`

**Key Components**:

1. **Model Management**
   - Global model loading on startup using `load_flowmo_model`
   - Pre-initialize LPIPS models (Alex & VGG)
   - Handle CUDA device management

2. **Image Processing Pipeline**
   - Accept PIL Image or base64 encoded images
   - Use existing `process_image` function
   - Add new function to apply AWGN noise directly to images (not just latent codes)

3. **New Functions Needed**:
   ```python
   def apply_awgn_to_image(image_tensor, noise_level):
       """Apply AWGN noise directly to image tensor"""
       # Convert noise_level to SNR and add noise to image
   
   def run_full_inference(model, image, noise_level, lpips_alex, lpips_vgg):
       """Complete inference pipeline with both noisy and reconstructed images"""
       # 1. Create noisy image
       # 2. Run model reconstruction
       # 3. Calculate metrics for both comparisons
       # 4. Return all results
   ```

4. **FastAPI Endpoints**:
   - `POST /process_image` - Main endpoint for image processing
   - `GET /health` - Health check endpoint
   - `GET /model_info` - Model information endpoint

#### **Phase 2: Frontend Implementation**

**Location**: `/home/utka/proj/semacomm/frontend/`

**Structure**:
```
frontend/
├── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
├── requirements.txt (if using Python server)
└── README.md
```

**Frontend Components**:

1. **Image Upload Interface**
   - Drag-and-drop file upload
   - Image preview
   - Supported formats: JPG, PNG, WEBP

2. **Noise Level Control**
   - Range slider: -5 to +20
   - Real-time value display
   - Slider labels for clarity

3. **Results Display**
   - Three-panel layout: Original | Noisy | Reconstructed
   - Metrics display below each image
   - Responsive design for different screen sizes

4. **API Communication**
   - Async/await for API calls
   - Loading states and progress indicators
   - Error handling and user feedback

#### **Phase 3: Integration and Communication**

**API Request/Response Format**:

```json
// Request
{
  "image": "base64_encoded_image_data",
  "noise_level": 7,
  "image_format": "jpeg"
}

// Response
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
  "noise_level": 7
}
```

### **Technical Implementation Details**

#### **Backend Dependencies**:
```python
# Additional imports needed for api_service.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import io
from typing import Optional
import uvicorn
```

#### **Key Functions to Implement**:

1. **Enhanced Image Processing**:
   ```python
   def apply_awgn_to_image(image_tensor, noise_level):
       """Apply AWGN noise directly to image tensor (not latent space)"""
       # Similar to apply_awgn_noise but works on image pixels
       
   def process_image_with_noise(image_tensor, noise_level):
       """Create noisy version of input image"""
       # Apply noise and return noisy image tensor
   ```

2. **Comprehensive Inference Pipeline**:
   ```python
   def run_complete_inference(model, image, noise_level, lpips_alex, lpips_vgg):
       """
       Complete pipeline:
       1. Original image
       2. Add noise to create noisy image
       3. Reconstruct using model
       4. Calculate metrics between: original-noisy, original-reconstructed
       5. Return all images and metrics
       """
   ```

3. **Image Conversion Utilities**:
   ```python
   def tensor_to_base64(tensor):
       """Convert tensor to base64 string"""
       
   def base64_to_tensor(base64_string):
       """Convert base64 string to tensor"""
   ```

#### **Frontend JavaScript Architecture**:

```javascript
// main.js structure
class ImageProcessor {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.initializeElements();
        this.setupEventListeners();
    }
    
    async uploadImage(file) {
        // Convert to base64 and prepare for API
    }
    
    async processImage(imageData, noiseLevel) {
        // Send to API and handle response
    }
    
    displayResults(response) {
        // Update UI with images and metrics
    }
}
```

### **Configuration Requirements**

1. **Model Files**: Ensure flowmo_lo.pth is accessible
2. **CUDA Setup**: GPU availability for inference
3. **Config Files**: `base.yaml` for model configuration
4. **Dependencies**: All packages from `infer.py` imports

### **Development Workflow**

1. **Step 1**: Create API service with model loading
2. **Step 2**: Implement noise addition for images (not just latent codes)
3. **Step 3**: Create comprehensive inference pipeline
4. **Step 4**: Build FastAPI endpoints
5. **Step 5**: Develop frontend interface
6. **Step 6**: Test integration and optimize performance

### **Performance Considerations**

1. **Model Loading**: Load once on startup, keep in memory
2. **GPU Memory**: Manage CUDA memory for concurrent requests
3. **Image Processing**: Optimize tensor operations
4. **Caching**: Consider caching frequently used results
5. **Async Processing**: Use async/await for non-blocking operations

This plan provides a comprehensive roadmap for implementing the FlowMo API service and frontend, leveraging the existing inference pipeline while adding the requested noise application and metric calculation features.