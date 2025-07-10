# FlowMo Frontend

This is the web frontend for the FlowMo image reconstruction service. It provides an intuitive interface for uploading images, applying noise, and viewing reconstruction results.

## Features

- **Drag & Drop Image Upload**: Easy image upload with preview
- **Noise Level Control**: Adjustable noise level from -5 to +20 PSNR
- **Three-Panel Results**: View original, noisy, and reconstructed images side by side
- **Comprehensive Metrics**: Display PSNR, SSIM, and LPIPS metrics
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Processing**: Live processing with loading indicators

## Structure

```
frontend/
├── index.html          # Main HTML interface
├── static/
│   ├── css/
│   │   └── style.css   # Styling and responsive design
│   ├── js/
│   │   └── main.js     # JavaScript functionality
│   └── images/         # Static images (if needed)
└── README.md           # This file
```

## Usage

1. **Start the API Service**: Ensure the FlowMo API service is running on `http://localhost:8000`

2. **Serve the Frontend**: You can serve the frontend using any web server:
   
   ```bash
   # Using Python's built-in server
   cd frontend
   python -m http.server 8080
   
   # Using Node.js serve package
   npx serve .
   
   # Or open index.html directly in a browser
   ```

3. **Access the Interface**: Open your browser and navigate to the served address (e.g., `http://localhost:8080`)

## How to Use

1. **Upload an Image**: 
   - Drag and drop an image onto the upload area, or
   - Click "Choose Image" to select a file

2. **Adjust Noise Level**:
   - Use the slider to set the noise level (-5 to +20 PSNR)
   - Higher values = less noise
   - Lower values = more noise

3. **Process Image**:
   - Click "Process Image" to start the reconstruction
   - Wait for the processing to complete

4. **View Results**:
   - Original image (as uploaded)
   - Noisy image (with applied AWGN noise)
   - Reconstructed image (FlowMo output)
   - Metrics comparing original vs noisy and original vs reconstructed

## API Integration

The frontend communicates with the FlowMo API service using:

- **POST /process_image**: Main processing endpoint
- **GET /health**: Health check to verify API availability
- **GET /model_info**: Get model information

## Supported Image Formats

- JPEG
- PNG
- WEBP
- Maximum file size: 10MB

## Browser Compatibility

- Modern browsers with ES6+ support
- Chrome, Firefox, Safari, Edge
- Mobile browsers

## Error Handling

The frontend includes comprehensive error handling for:

- Network connectivity issues
- API service unavailability
- Invalid image formats
- File size limitations
- Processing errors

## Performance Considerations

- Images are automatically resized to 256x256 for processing
- Base64 encoding is used for image transmission
- Loading indicators provide user feedback during processing
- Responsive design ensures good performance on all devices
