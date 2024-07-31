from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import io

app = Flask(__name__)

# Load the trained autoencoder model
autoencoder = tf.keras.models.load_model('autoencoder_model.h5')

# Define a route for the default URL, which loads the upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Define a route for handling the uploaded file
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Preprocess the image
        image = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict using the model
        deblurred_image = autoencoder.predict(image)
        deblurred_image = deblurred_image[0]
        
        # Convert the result to an image
        deblurred_image = (deblurred_image * 255).astype(np.uint8)
        deblurred_image = Image.fromarray(deblurred_image)
        
        # Save the deblurred image
        deblurred_image_path = os.path.join('outputs', filename)
        deblurred_image.save(deblurred_image_path)
        
        return send_file(deblurred_image_path, mimetype='image/jpeg')

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    app.run(debug=True)
