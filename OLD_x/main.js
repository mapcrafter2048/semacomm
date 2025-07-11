/**
 * FlowMo Frontend JavaScript
 * Handles image upload, API communication, and results display
 */

class ImageProcessor {
    constructor() {
        this.apiUrl = 'http://localhost:4040';
        this.currentImageData = null;
        this.originalImageSize = 0; // Store original image size in bytes
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.removeBtn = document.getElementById('removeBtn');

        // Control elements
        this.noiseSlider = document.getElementById('noiseSlider');
        this.noiseValue = document.getElementById('noiseValue');
        this.processBtn = document.getElementById('processBtn');

        // Results elements
        this.resultsSection = document.getElementById('resultsSection');
        this.originalImg = document.getElementById('originalImg');
        this.noisyImg = document.getElementById('noisyImg');
        this.reconstructedImg = document.getElementById('reconstructedImg');

        // Metric elements
        this.noisyPsnr = document.getElementById('noisyPsnr');
        this.noisySsim = document.getElementById('noisySsim');
        this.noisyLpipsAlex = document.getElementById('noisyLpipsAlex');
        this.noisyLpipsVgg = document.getElementById('noisyLpipsVgg');

        this.reconstructedPsnr = document.getElementById('reconstructedPsnr');
        this.reconstructedSsim = document.getElementById('reconstructedSsim');
        this.reconstructedLpipsAlex = document.getElementById('reconstructedLpipsAlex');
        this.reconstructedLpipsVgg = document.getElementById('reconstructedLpipsVgg');

        // Info elements
        this.processingTime = document.getElementById('processingTime');
        this.appliedNoiseLevel = document.getElementById('appliedNoiseLevel');

        // Size elements
        this.originalSize = document.getElementById('originalSize');
        this.noisySize = document.getElementById('noisySize'); 
        this.reconstructedSize = document.getElementById('reconstructedSize');
        this.compressionRatio = document.getElementById('compressionRatio');

        // Loading and error elements
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.errorMessage = document.getElementById('errorMessage');
        this.errorText = document.getElementById('errorText');
    }

    setupEventListeners() {
        // File input change - remove any existing listeners first
        this.imageInput.removeEventListener('change', this.boundHandleFileSelect);
        this.boundHandleFileSelect = (e) => this.handleFileSelect(e);
        this.imageInput.addEventListener('change', this.boundHandleFileSelect);

        // Drag and drop - remove existing listeners first
        this.uploadArea.removeEventListener('click', this.boundUploadClick);
        this.uploadArea.removeEventListener('dragover', this.boundHandleDragOver);
        this.uploadArea.removeEventListener('dragleave', this.boundHandleDragLeave);
        this.uploadArea.removeEventListener('drop', this.boundHandleDrop);

        this.boundUploadClick = () => this.imageInput.click();
        this.boundHandleDragOver = (e) => this.handleDragOver(e);
        this.boundHandleDragLeave = (e) => this.handleDragLeave(e);
        this.boundHandleDrop = (e) => this.handleDrop(e);

        this.uploadArea.addEventListener('click', this.boundUploadClick);
        this.uploadArea.addEventListener('dragover', this.boundHandleDragOver);
        this.uploadArea.addEventListener('dragleave', this.boundHandleDragLeave);
        this.uploadArea.addEventListener('drop', this.boundHandleDrop);

        // Remove button
        this.removeBtn.addEventListener('click', () => this.removeImage());

        // Noise slider
        this.noiseSlider.addEventListener('input', (e) => this.updateNoiseValue(e));

        // Process button
        this.processBtn.addEventListener('click', () => this.processImage());

        // Check API health on load
        this.checkApiHealth();
    }

    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy' && data.model_loaded && data.lpips_loaded) {
                console.log('API is healthy and ready');
            } else {
                this.showError('API is not ready. Please ensure the FlowMo service is running.');
            }
        } catch (error) {
            console.warn('API health check failed:', error);
            this.showError('Cannot connect to FlowMo API. Please ensure the service is running on http://localhost:4040');
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('Image file is too large. Please select an image smaller than 10MB.');
            return;
        }

        this.loadImage(file);
    }

    loadImage(file) {
        const reader = new FileReader();
        
        // Store original file size
        this.originalImageSize = file.size;
        
        reader.onload = (e) => {
            this.currentImageData = e.target.result.split(',')[1]; // Remove data URL prefix
            this.previewImg.src = e.target.result;
            
            // Ensure proper image display
            this.previewImg.onload = () => {
                this.uploadArea.style.display = 'none';
                this.imagePreview.style.display = 'block';
                this.processBtn.disabled = false;
                this.hideResults();
            };
        };
        
        reader.onerror = () => {
            this.showError('Failed to load image. Please try again.');
        };
        
        reader.readAsDataURL(file);
    }

    removeImage() {
        this.currentImageData = null;
        this.originalImageSize = 0;
        this.previewImg.src = '';
        this.uploadArea.style.display = 'block';
        this.imagePreview.style.display = 'none';
        this.processBtn.disabled = true;
        this.imageInput.value = '';
        this.hideResults();
    }

    updateNoiseValue(e) {
        this.noiseValue.textContent = parseFloat(e.target.value).toFixed(1);
    }

    // Helper function to calculate base64 size in bytes
    calculateBase64Size(base64String) {
        // Base64 encoding increases size by ~33%
        // Remove data URL prefix if present
        const base64Data = base64String.split(',').pop();
        return Math.ceil(base64Data.length * 3 / 4);
    }

    // Helper function to format bytes to KB
    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // Helper function to format bytes to Kb (kilobits)
    formatKilobits(bytes, decimals = 1) {
        const bits = bytes * 8;
        const kilobits = bits / 1000; // Using 1000 for network transmission
        return parseFloat(kilobits.toFixed(decimals)) + ' Kb';
    }

    async processImage() {
        if (!this.currentImageData) {
            this.showError('Please select an image first.');
            return;
        }

        const noiseLevel = parseFloat(this.noiseSlider.value);
        
        try {
            this.showLoading();
            
            const requestData = {
                image: this.currentImageData,
                noise_level: noiseLevel,
                image_format: "png"
            };

            const response = await fetch(`${this.apiUrl}/process_image`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}`);
            }

            const result = await response.json();
            this.displayResults(result);
            
        } catch (error) {
            console.error('Processing error:', error);
            this.showError(`Failed to process image: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        // Display images
        this.originalImg.src = `data:image/png;base64,${result.original_image}`;
        this.noisyImg.src = `data:image/png;base64,${result.noisy_image}`;
        this.reconstructedImg.src = `data:image/png;base64,${result.reconstructed_image}`;

        // Calculate image sizes
        const originalSizeBytes = this.originalImageSize;
        const noisySizeBytes = this.calculateBase64Size(result.noisy_image);
        const reconstructedSizeBytes = this.calculateBase64Size(result.reconstructed_image);



        // Display size information
        if (this.originalSize) {
            this.originalSize.textContent = `Size: ${this.formatKilobits(originalSizeBytes)}`;
        }
        if (this.noisySize) {
            this.noisySize.textContent = `Transmission Size: ${this.formatKilobits(noisySizeBytes)}`;
        }
        if (this.reconstructedSize) {
            const transmissionKb = this.formatKilobits(reconstructedSizeBytes);
            this.reconstructedSize.textContent = `Transmission Size: ${transmissionKb}`;
        }

        // Calculate and display compression ratio
        if (this.compressionRatio) {
            const ratio = (reconstructedSizeBytes / originalSizeBytes) * 100;
            this.compressionRatio.textContent = `Transmitted image is ${ratio.toFixed(1)}% of original image`;
        }

        // Display noisy metrics
        const noisyMetrics = result.metrics.original_vs_noisy;
        this.noisyPsnr.textContent = noisyMetrics.psnr.toFixed(2);
        this.noisySsim.textContent = noisyMetrics.ssim.toFixed(4);
        this.noisyLpipsAlex.textContent = noisyMetrics.lpips_alex.toFixed(4);
        this.noisyLpipsVgg.textContent = noisyMetrics.lpips_vgg.toFixed(4);

        // Display reconstructed metrics
        const reconstructedMetrics = result.metrics.original_vs_reconstructed;
        this.reconstructedPsnr.textContent = reconstructedMetrics.psnr.toFixed(2);
        this.reconstructedSsim.textContent = reconstructedMetrics.ssim.toFixed(4);
        this.reconstructedLpipsAlex.textContent = reconstructedMetrics.lpips_alex.toFixed(4);
        this.reconstructedLpipsVgg.textContent = reconstructedMetrics.lpips_vgg.toFixed(4);

        // Display processing info
        this.processingTime.textContent = `${result.processing_time.toFixed(2)} seconds`;
        this.appliedNoiseLevel.textContent = result.noise_level.toFixed(1);

        // Show results section
        this.resultsSection.style.display = 'block';
        
        // Smooth scroll to results
        this.resultsSection.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
        this.processBtn.disabled = true;
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
        this.processBtn.disabled = false;
    }

    showError(message) {
        this.errorText.textContent = message;
        this.errorMessage.style.display = 'flex';
    }

    hideError() {
        this.errorMessage.style.display = 'none';
    }
}

// Global function for error close button
function hideError() {
    if (window.imageProcessor) {
        window.imageProcessor.hideError();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.imageProcessor = new ImageProcessor();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    // Clean up resources if needed
    if (window.imageProcessor) {
        window.imageProcessor.currentImageData = null;
    }
});
