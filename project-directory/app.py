from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def gaussian_filter(img, gaus_kernel_size=7, sigma=1, muu=0):
    x, y = np.meshgrid(np.linspace(-1, 1, gaus_kernel_size),
                       np.linspace(-1, 1, gaus_kernel_size))
    dst = np.sqrt(x**2 + y**2)
    normal = 1 / (((2 * np.pi)**0.5) * sigma)
    gauss = np.exp(-((dst - muu)**2 / (2.0 * sigma**2))) * normal
    gauss = np.pad(gauss, [(0, img.shape[0] - gauss.shape[0]), (0, img.shape[1] - gauss.shape[1])], 'constant')
    
    return gauss

def fft_deblur_channel(img, gaus_kernel_size=7, kernel_sigma=5, factor='wiener', const=0.002):
    
    gauss = gaussian_filter(img, gaus_kernel_size, kernel_sigma)
    
    img_fft = np.fft.fft2(img)
    gauss_fft = np.fft.fft2(gauss)
    weiner_factor = 1 / (1 + (const / np.abs(gauss_fft)**2))
    if factor != 'wiener':
        weiner_factor = factor
    recon = img_fft / gauss_fft
    recon *= weiner_factor
    recon = np.abs(np.fft.ifft2(recon))

    recon = cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX)
    
    return recon

def deblur_image(image, gaus_kernel_size=7, kernel_sigma=5, factor='wiener', const=0.5, brightness_factor=1.5):
    
    if len(image.shape) == 2:
        recon = fft_deblur_channel(image, gaus_kernel_size, kernel_sigma, factor, const)
    else:
        channels = cv2.split(image)
        recon_channels = [fft_deblur_channel(channel, gaus_kernel_size, kernel_sigma, factor, const) for channel in channels]
        recon = cv2.merge(recon_channels)

    recon *= brightness_factor
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    
    return recon

def read_image(path):
    im = cv2.imread(path, -1)
    
    return im

def blur_input_image(normal_image_path, blur_kernel_size=(7, 7), sigma=5):
    
    normal_image = read_image(normal_image_path)
    
    im_blur = cv2.GaussianBlur(normal_image, blur_kernel_size, sigma)
    
    return im_blur

def deblur_artificially_blurred_image(normal_image_path, blur_kernel_size=(7, 7), gaus_kernel_size=7, kernel_sigma=5, factor='wiener', const=0.002, brightness_factor = 1.5):
    
    artifically_blurred_image = blur_input_image(normal_image_path, blur_kernel_size, kernel_sigma)
    
    if len(artifically_blurred_image.shape) == 2:
        recon = fft_deblur_channel(artifically_blurred_image, gaus_kernel_size, kernel_sigma, factor, const)
    else:
        channels = cv2.split(artifically_blurred_image)
        recon_channels = [fft_deblur_channel(channel, gaus_kernel_size, kernel_sigma, factor, const) for channel in channels]
        recon = cv2.merge(recon_channels)

    recon *= brightness_factor
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    
    return recon

def deblur_originally_blurred_image(originally_blurred_image_path, gaus_kernel_size=7, kernel_sigma=5, factor='wiener', const=0.002, brightness_factor = 1.5):
    
    originally_blurred_image = read_image(originally_blurred_image_path)
    
    if len(originally_blurred_image.shape) == 2:
        recon = fft_deblur_channel(originally_blurred_image, gaus_kernel_size, kernel_sigma, factor, const)
    else:
        channels = cv2.split(originally_blurred_image)
        recon_channels = [fft_deblur_channel(channel, gaus_kernel_size, kernel_sigma, factor, const) for channel in channels]
        recon = cv2.merge(recon_channels)

    recon *= brightness_factor
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    
    return recon

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'process_type' not in request.form:
        return 'No file part or process type selected'
    file = request.files['file']
    process_type = request.form['process_type']
    const = float(request.form['const'])
    brightness_factor = float(request.form['brightness_factor'])
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if process_type == 'normal_image':
            original_image = read_image(filepath)
            blurred_image = blur_input_image(filepath)
            deblurred_image = deblur_artificially_blurred_image(filepath, const=const, brightness_factor=brightness_factor)
            images = {'original': original_image, 'blurred': blurred_image, 'deblurred': deblurred_image}
        elif process_type == 'originally_blurred':
            original_image = cv2.imread(filepath)
            deblurred_image = deblur_originally_blurred_image(filepath, 7, 5, 'wiener', const=const, brightness_factor=brightness_factor)
            images = {'original': original_image, 'deblurred': deblurred_image}
        else:
            return 'Invalid process type selected'

        outputs = {}
        for key, image in images.items():
            _, buffer = cv2.imencode('.png', image)
            io_buf = io.BytesIO(buffer)
            encoded_image = base64.b64encode(io_buf.getvalue()).decode('utf-8')
            outputs[key] = encoded_image

        response = '''
        <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f0f0f0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    display: flex;
                    justify-content: space-around;
                    width: 90%;
                    max-width: 1000px;
                }
                h3 {
                    text-align: center;
                }
                .image-container {
                    flex: 1;
                    text-align: center;
                }
                img {
                    max-width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <div class="container">
        '''
        for key, encoded_image in outputs.items():
            response += f'''
                <div class="image-container">
                    <h3>{key.capitalize()} Image</h3>
                    <img src="data:image/png;base64,{encoded_image}"/>
                </div>
            '''
        response += '''
            </div>
        </body>
        </html>
        '''

        return response

if __name__ == '__main__':
    app.run(debug=True)