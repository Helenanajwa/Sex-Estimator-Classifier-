import os
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

routes = Blueprint('routes', __name__)

# Load models
model_ap = load_model('model/best_model_AP.h5')
model_lateral = load_model('model/best_model_Lateral.h5')

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@routes.route('/')
def index():
    return render_template('index.html')

@routes.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Prepare image
        img_array = prepare_image(filepath)

        # Determine which model to use based on form input
        view_type = request.form.get('view_type', 'AP').lower()

        if view_type == 'ap':
            prediction = model_ap.predict(img_array)
        else:
            prediction = model_lateral.predict(img_array)

        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })

    return jsonify({'error': 'Invalid file type'}), 400
