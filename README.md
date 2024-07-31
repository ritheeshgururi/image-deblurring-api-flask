# Image Deblurring API with Flask

A web-based application that implements two approaches to remove blur from images using **1. Deep Learning (Auto Encoder) and 2. Classical Image Processing (Wiener Filter)**.

This Flask-based API can:
- Remove blur from images using Wiener filtering
- Remove blur using deep learning (autoencoder)
- Handle both artificially blurred and naturally blurred images
- Process both color and grayscale images

## Directory Structure

```
image-deblurring-api-flask/
├── wiener-filter-deblurring/
│   └── image_deblurring_using_wiener_filtering_latest.py
├── auto-encoder-deblurring/
│   ├── app2.py
│   ├── image_deblurring_using_autoencoders.py
│   └── templates/
└── project-directory/
    └── app.py
```

## Approach

### 1. Autoencoder Approach
- Processes images to 128x128 pixel size
- Trained on pairs of clear and blurred images
- Automatically determines best deblurring parameters

### 2. Wiener Filter Approach
- Takes a blurred image as input
- Applies Fourier transformation to the image
- Uses Wiener filtering algorithm to remove blur
- Adjusts brightness and normalizes the output
- Parameters can be customized:
  - Kernel size (default: 7x7)
  - Sigma value (default: 5)
  - Brightness factor (default: 1.5)
  - Wiener filter constant (default: 0.002)

### 3. Web Interface
- Simple upload interface with error handling for invalid uploads
- Supports multiple image formats (JPG, JPEG, PNG)
- Shows original, blurred (if artificial), and deblurred images

## How to use

1. For Artificial Blur:
   - Upload a clear image
   - Applies Gaussian blur
   - Removes the blur
   - Compare original, blurred, and deblurred results

2. For Natural Blur:
   - Upload an already blurred image
   - Choose processing method (Wiener or Autoencoder)
   - Preprocesses and deblurs the image
   - Outputs deblurred image

## Implementation

### Image Processing Steps:
1. Image Upload
   - File validation
   - Format checking

2. Preprocessing
   - Image loading
   - Size normalization (if needed)
   - Color handling (RGB/Grayscale)

3. Deblurring Process
   - Wiener Filter:
     - FFT transformation
     - Gaussian kernel application
     - Wiener filtering
     - Inverse FFT
   - Autoencoder:
     - Image scaling
     - Autoencoder processing
     - Result reconstruction

4. Post-processing
   - Brightness adjustment
   - Pixel value normalization
   - Format conversion

### Key Functions:

1. Wiener Filter:
   ```python
   deblur_image()
   fft_deblur_channel()
   blur_input_image()
   ```

2. Autoencoder:
   ```python
   autoencoder.predict()
   preprocess_image()
   ```

## API Implementation

### Flask API Endpoints

1. Main Application (`app.py`):
```python
@app.route('/')
# Main landing page
# Returns: HTML template with upload form

@app.route('/upload', methods=['POST'])
# Handles image upload and processing
# Parameters:
#   - file: Image file (multipart/form-data)
#   - process_type: 'normal_image' or 'originally_blurred'
#   - const: Wiener filter constant (float)
#   - brightness_factor: Brightness adjustment (float)
# Returns: THML page with processed images
```

2. Autoencoder Application (`app2.py`):
```python
@app.route('/')
# Autoencoder upload page
# Returns: HTML template for autoencoder processing

@app.route('/upload', methods=['POST'])
# Handles autoencoder-based deblurring
# Parameters:
#   - file: Image file (multipart/form-data)
# Returns: Processed image file
```