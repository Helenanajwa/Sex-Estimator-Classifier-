from flask import Blueprint, render_template, redirect, url_for, session, request, jsonify, current_app
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import logging
import random

main = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

# Optional: Set random seed for consistent predictions
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Lazy-loaded models (cache them so they load only once)
model_lat = None
model_ap = None

def allowed_file(filename, app):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    img = tf.keras.utils.load_img(
        img_path,
        color_mode='rgb',
        target_size=target_size
    )
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_lateral_model():
    """Lazy load Lateral view model"""
    global model_lat
    if model_lat is None:
        logger.info("Loading Lateral view model...")
        model_lat = tf.keras.models.load_model('model.py/best_model.h5')
    return model_lat

def get_ap_model():
    """Lazy load AP view model"""
    global model_ap
    if model_ap is None:
        logger.info("Loading AP view model...")
        model_ap = tf.keras.models.load_model('model.py/best_model_AP.h5')
    return model_ap

# ----------------- MAIN PAGES -----------------
@main.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    return render_template('index.html')

@main.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    return render_template('home.html')

@main.route('/confusion-matrix')
def model_performance():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    return render_template('model_performance.html')

@main.route('/research')
def research():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    return render_template('research.html')

@main.route('/collaborate')
def collaborate():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    return render_template('collaborate.html')

# ----------------- ANALYSIS ROUTE -----------------
@main.route('/analyze', methods=['POST'])
def analyze():
    view = request.form.get('view')
    file = request.files.get('xray_image')

    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    if not view:
        return jsonify({'success': False, 'error': 'No view selected'}), 400

    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    try:
        # Directly run the selected model
        processed = preprocess_image(filepath, target_size=(224, 224))

        if view == 'ap':
            model_ap_loaded = get_ap_model()
            pred = model_ap_loaded.predict(processed)
        elif view == 'lateral':
            model_lat_loaded = get_lateral_model()
            pred = model_lat_loaded.predict(processed)
        else:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Invalid view type'}), 400

        gender = 'Female' if pred[0][0] > 0.5 else 'Male'
        confidence = round(pred[0][0]*100 if gender == 'Female' else (1-pred[0][0])*100, 1)

        return jsonify({
            'success': True,
            'gender': gender,
            'confidence': confidence,
            'view': view
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
