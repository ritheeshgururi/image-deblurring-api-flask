from flask import Flask, request, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/blurred'
UNBLURRED_FOLDER = 'static/unblurred'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UNBLURRED_FOLDER'] = UNBLURRED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Check if corresponding unblurred image exists
        if os.path.exists(os.path.join(app.config['UNBLURRED_FOLDER'], filename)):
            return render_template('display.html', 
                                   blurred_image=os.path.join('static/blurred', filename), 
                                   unblurred_image=os.path.join('static/unblurred', filename))
        else:
            abort(404, description="Unblurred image not found")

if __name__ == "__main__":
    app.run(debug=True)