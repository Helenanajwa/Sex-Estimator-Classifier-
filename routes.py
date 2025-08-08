from flask import Blueprint, render_template, redirect, url_for, session, request, jsonify, current_app
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import logging
import random

main = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    logger.info("No GPU detected, using CPU")
else:
    logger.warning(f"GPU detected but disabled: {physical_devices}")

# Optional: Set random seed for consistent predictions
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Verify model files exist
model_paths = ['model/best_model.h5', 'model/best_model_AP.h5', 'model/xray_classifier.h5']
for path in model_paths:
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        raise FileNotFoundError(f"Model file not found: {path}")

# Load models once for efficiency
try:
    model_lat = tf.keras.models.load_model('model/best_model.h5')
    model_ap = tf.keras.models.load_model('model/best_model_AP.h5')
    xray_classifier = tf.keras.models.load_model('model/xray_classifier.h5')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

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

def is_xray_image(img_path):
    """Check if the uploaded image is an X-ray using the classifier"""
    processed_img = preprocess_image(img_path, target_size=(128, 128))
    prediction = xray_classifier.predict(processed_img)
    is_xray = prediction[0][0] > 0.7
    confidence = float(prediction[0][0]) if is_xray else float(1 - prediction[0][0])
    return is_xray, confidence

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
    logger.info("Starting analyze route")
    ap_file = request.files.get('xray_ap')
    lat_files = [f for k, f in request.files.items() if k.startswith('xray_lat_') and f.filename != '']
    logger.info(f"Received AP file: {ap_file.filename if ap_file else None}, LAT files: {[f.filename for f in lat_files]}")

    # Store temporary file paths
    temp_files = []

    # --- Step 1: Validate all files are X-rays ---
    if ap_file and ap_file.filename != '':
        filepath_ap = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(ap_file.filename))
        ap_file.save(filepath_ap)
        temp_files.append(filepath_ap)
        is_xray, confidence = is_xray_image(filepath_ap)
        logger.info(f"AP X-ray check: is_xray={is_xray}, confidence={confidence}")
        if not is_xray:
            cleanup_files(temp_files)
            return jsonify({'success': False, 'error': 'Please upload the specific type of AP view X-ray.'}), 400

    for lf in lat_files:
        filepath_lat = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(lf.filename))
        lf.save(filepath_lat)
        temp_files.append(filepath_lat)
        is_xray, confidence = is_xray_image(filepath_lat)
        logger.info(f"LAT X-ray check: is_xray={is_xray}, confidence={confidence}")
        if not is_xray:
            cleanup_files(temp_files)
            return jsonify({'success': False, 'error': 'Please upload the specific type of Lateral view X-ray.'}), 400

    # --- Step 2: Run predictions only if all passed ---
    ap_result = None
    lat_result = None

    if ap_file and ap_file.filename != '':
        processed = preprocess_image(filepath_ap, target_size=(224, 224))
        pred = model_ap.predict(processed)
        gender = 'Female' if pred[0][0] > 0.5 else 'Male'
        conf = round(pred[0][0]*100 if gender == 'Female' else (1-pred[0][0])*100, 1)
        ap_result = {'gender': gender, 'confidence': conf}
        logger.info(f"AP prediction: gender={gender}, confidence={conf}")

    if lat_files:
        confs = []
        genders = []
        for lf in lat_files:
            processed = preprocess_image(os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(lf.filename)), target_size=(224, 224))
            pred = model_lat.predict(processed)
            gender = 'Female' if pred[0][0] > 0.5 else 'Male'
            conf = round(pred[0][0]*100 if gender == 'Female' else (1-pred[0][0])*100, 1)
            confs.append(conf)
            genders.append(gender)
            logger.info(f"LAT prediction: gender={gender}, confidence={conf}")
        final_gender = max(set(genders), key=genders.count)
        final_conf = round(sum(confs)/len(confs), 1)
        lat_result = {'gender': final_gender, 'confidence': final_conf}
        logger.info(f"LAT final result: gender={final_gender}, confidence={final_conf}")

    # --- Step 3: Cleanup ---
    cleanup_files(temp_files)

    return jsonify({'success': True, 'AP': ap_result, 'LAT': lat_result})

def cleanup_files(file_list):
    """Remove temporary uploaded files"""
    for file_path in file_list:
        if os.path.exists(file_path):
            os.remove(file_path)
