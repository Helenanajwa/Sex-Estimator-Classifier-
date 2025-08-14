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

# Global model variables
model_ap = None
model_lat = None

def load_models():
    """Load TensorFlow models at startup"""
    global model_ap, model_lat
    ap_model_path = 'model.py/best_model_AP.h5'
    lat_model_path = 'model.py/best_model.h5'
    if not os.path.exists(ap_model_path):
        logger.error(f"AP model file not found: {ap_model_path}")
        raise FileNotFoundError(f"AP model file not found: {ap_model_path}")
    if not os.path.exists(lat_model_path):
        logger.error(f"Lateral model file not found: {lat_model_path}")
        raise FileNotFoundError(f"Lateral model file not found: {lat_model_path}")
    try:
        logger.info("Loading AP model at startup")
        global model_ap
        model_ap = tf.keras.models.load_model(ap_model_path)
        logger.info("AP model loaded successfully")
        logger.info("Loading Lateral model at startup")
        global model_lat
        model_lat = tf.keras.models.load_model(lat_model_path)
        logger.info("Lateral model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

# Allowed extensions
def allowed_file(filename, app):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    logger.info(f"Preprocessing image: {img_path}")
    img = tf.keras.utils.load_img(
        img_path,
        color_mode='rgb',
        target_size=target_size
    )
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return tf.expand_dims(img_array, axis=0)

def cleanup_files(file_list):
    """Remove temporary uploaded files"""
    for file_path in file_list:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up file {file_path}: {str(e)}")

# ----------------- MAIN PAGES -----------------
@main.route('/')
def index():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))  # Fallback to home route
    return render_template('index.html')

@main.route('/home')
def home():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))  # Fallback to home route
    return render_template('home.html')

@main.route('/confusion-matrix')
def model_performance():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))  # Fallback to home route
    return render_template('model_performance.html')

@main.route('/research')
def research():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))  # Fallback to home route
    return render_template('research.html')

@main.route('/collaborate')
def collaborate():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))  # Fallback to home route
    return render_template('collaborate.html')

# ----------------- ANALYSIS ROUTE -----------------
@main.route('/analyze', methods=['POST'])
@main.route('/analyze/', methods=['POST'])  # Handle trailing slash
def analyze():
    logger.info("Starting analyze route")
    ap_file = request.files.get('xray_ap')
    lat_files = [f for k, f in request.files.items() if k.startswith('xray_lat_') and f.filename != '']
    logger.info(f"Received AP file: {ap_file.filename if ap_file else None}, LAT files: {[f.filename for f in lat_files]}")

    temp_files = []
    ap_result, lat_result = None, None

    # Validate and save AP file
    if ap_file and ap_file.filename != '':
        if not allowed_file(ap_file.filename, current_app):
            logger.error("Invalid file type for AP image")
            return jsonify({'success': False, 'error': 'Invalid file type for AP image'}), 400
        filepath_ap = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(ap_file.filename))
        try:
            ap_file.save(filepath_ap)
            temp_files.append(filepath_ap)
            logger.info(f"Saved AP file: {filepath_ap}")
        except Exception as e:
            logger.error(f"Failed to save AP file: {str(e)}")
            return jsonify({'success': False, 'error': f'Failed to save AP file: {str(e)}'}), 500

    # Validate and save Lateral files
    for lf in lat_files:
        if not allowed_file(lf.filename, current_app):
            logger.error(f"Invalid file type for Lateral image: {lf.filename}")
            return jsonify({'success': False, 'error': f'Invalid file type for Lateral image: {lf.filename}'}), 400
        filepath_lat = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(lf.filename))
        try:
            lf.save(filepath_lat)
            temp_files.append(filepath_lat)
            logger.info(f"Saved Lateral file: {filepath_lat}")
        except Exception as e:
            logger.error(f"Failed to save Lateral file: {str(e)}")
            cleanup_files(temp_files)
            return jsonify({'success': False, 'error': f'Failed to save Lateral file: {str(e)}'}), 500

    try:
        # AP prediction
        if ap_file and ap_file.filename != '':
            try:
                processed = preprocess_image(filepath_ap, target_size=(224, 224))
                pred = model_ap.predict(processed, verbose=0)
                gender = 'Female' if pred[0][0] > 0.5 else 'Male'
                conf = round(pred[0][0] * 100 if gender == 'Female' else (1 - pred[0][0]) * 100, 1)
                ap_result = {'gender': gender, 'confidence': conf}
                logger.info(f"AP prediction: gender={gender}, confidence={conf}")
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.error(f"AP prediction failed: {str(e)}")
                cleanup_files(temp_files)
                return jsonify({'success': False, 'error': f'AP prediction failed: {str(e)}'}), 500

        # Lateral prediction
        if lat_files:
            try:
                confs, genders = [], []
                for lf in lat_files:
                    filepath_lat = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(lf.filename))
                    processed = preprocess_image(filepath_lat, target_size=(224, 224))
                    pred = model_lat.predict(processed, verbose=0)
                    gender = 'Female' if pred[0][0] > 0.5 else 'Male'
                    conf = round(pred[0][0] * 100 if gender == 'Female' else (1 - pred[0][0]) * 100, 1)
                    confs.append(conf)
                    genders.append(gender)
                    logger.info(f"LAT prediction: gender={gender}, confidence={conf}")
                final_gender = max(set(genders), key=genders.count)
                final_conf = round(sum(confs) / len(confs), 1)
                lat_result = {'gender': final_gender, 'confidence': final_conf}
                logger.info(f"LAT final result: gender={final_gender}, confidence={final_conf}")
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.error(f"Lateral prediction failed: {str(e)}")
                cleanup_files(temp_files)
                return jsonify({'success': False, 'error': f'Lateral prediction failed: {str(e)}'}), 500

    finally:
        cleanup_files(temp_files)

    return jsonify({'success': True, 'AP': ap_result, 'LAT': lat_result})
