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

# Set random seed for consistent predictions
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Log current directory and contents for debugging
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Root directory contents: {os.listdir('.')}")
if os.path.exists('model.py'):
    logger.info(f"model.py directory contents: {os.listdir('model.py')}")
else:
    logger.warning("model.py directory not found")

# Initialize model variables
model_lat = None
model_ap = None

# Verify and load model files
model_paths = ['model.py/best_model.h5', 'model.py/best_model_AP.h5']
for path in model_paths:
    if not os.path.exists(path):
        logger.warning(f"Model file not found: {path}. Analysis route will return error if used.")
    else:
        logger.info(f"Model file found: {path}")

# Load models if available
try:
    if os.path.exists('model.py/best_model.h5'):
        model_lat = tf.keras.models.load_model('model.py/best_model.h5')
        logger.info("Lateral model loaded successfully")
    if os.path.exists('model.py/best_model_AP.h5'):
        model_ap = tf.keras.models.load_model('model.py/best_model_AP.h5')
        logger.info("AP model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}. Analysis route will return error if used.")

def allowed_file(filename, app):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    logger.info(f"Preprocessing image: {img_path}")
    img = tf.keras.utils.load_img(
        img_path,
        color_mode='rgb',
        target_size=target_size
    )
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

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
@main.route('/analyze/', methods=['POST'])  # Handle trailing slash
def analyze():
    logger.info("Starting analyze route")
    ap_file = request.files.get('xray_ap')
    lat_files = [f for k, f in request.files.items() if k.startswith('xray_lat_') and f.filename != '']
    logger.info(f"Received AP file: {ap_file.filename if ap_file else None}, LAT files: {[f.filename for f in lat_files]}")

    # Check if models are loaded
    if model_ap is None and ap_file:
        logger.error("AP model not loaded")
        return jsonify({'success': False, 'error': 'AP model not available'}), 500
    if model_lat is None and lat_files:
        logger.error("Lateral model not loaded")
        return jsonify({'success': False, 'error': 'Lateral model not available'}), 500

    # Store temporary file paths
    temp_files = []

    # --- Step 1: Save files ---
    if ap_file and ap_file.filename != '':
        filepath_ap = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(ap_file.filename))
        ap_file.save(filepath_ap)
        temp_files.append(filepath_ap)

    for lf in lat_files:
        filepath_lat = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(lf.filename))
        lf.save(filepath_lat)
        temp_files.append(filepath_lat)

    # --- Step 2: Run predictions ---
    ap_result = None
    lat_result = None

    if ap_file and ap_file.filename != '' and model_ap:
        try:
            processed = preprocess_image(filepath_ap, target_size=(224, 224))
            pred = model_ap.predict(processed)
            gender = 'Female' if pred[0][0] > 0.5 else 'Male'
            conf = round(pred[0][0]*100 if gender == 'Female' else (1-pred[0][0])*100, 1)
            ap_result = {'gender': gender, 'confidence': conf}
            logger.info(f"AP prediction: gender={gender}, confidence={conf}")
        except Exception as e:
            logger.error(f"AP prediction failed: {str(e)}")
            cleanup_files(temp_files)
            return jsonify({'success': False, 'error': f'AP prediction failed: {str(e)}'}), 500

    if lat_files and model_lat:
        try:
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
        except Exception as e:
            logger.error(f"LAT prediction failed: {str(e)}")
            cleanup_files(temp_files)
            return jsonify({'success': False, 'error': f'Lateral prediction failed: {str(e)}'}), 500

    # --- Step 3: Cleanup ---
    cleanup_files(temp_files)

    return jsonify({'success': True, 'AP': ap_result, 'LAT': lat_result})

def cleanup_files(file_list):
    """Remove temporary uploaded files"""
    for file_path in file_list:
        if os.path.exists(file_path):
            os.remove(file_path)
