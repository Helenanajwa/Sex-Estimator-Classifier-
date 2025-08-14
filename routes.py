from flask import Blueprint, render_template, redirect, url_for, session, request, jsonify, current_app
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import logging
import random
import gc

main = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging to reduce overhead
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Force CPU usage and optimize TensorFlow
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    logger.info("No GPU detected, using CPU")
else:
    logger.warning(f"GPU detected but disabled: {physical_devices}")

# Set random seed for consistent predictions
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_model():
    """Load the combined model on demand"""
    model_path = 'model.py/bestmodel(both).h5'
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        logger.info("Loading combined model")
        model = tf.keras.models.load_model(model_path)
        logger.info("Combined model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load combined model: {str(e)}")
        raise

def allowed_file(filename, app):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    logger.info(f"Preprocessing image: {img_path} with target_size={target_size}")
    img = tf.keras.utils.load_img(
        img_path,
        color_mode='rgb',
        target_size=target_size
    )
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    gc.collect()
    return img_array

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
            return redirect(url_for('main.home'))
    return render_template('index.html')

@main.route('/home')
def home():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))
    return render_template('home.html')

@main.route('/confusion-matrix')
def model_performance():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))
    return render_template('model_performance.html')

@main.route('/research')
def research():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))
    return render_template('research.html')

@main.route('/collaborate')
def collaborate():
    if not session.get('logged_in'):
        try:
            return redirect(url_for('auth.login'))
        except Exception as e:
            logger.error(f"Failed to redirect to auth.login: {str(e)}")
            return redirect(url_for('main.home'))
    return render_template('collaborate.html')

# ----------------- ANALYSIS ROUTE -----------------
@main.route('/analyze', methods=['POST'])
@main.route('/analyze/', methods=['POST'])
def analyze():
    logger.info("Starting analyze route")
    xray_image = request.files.get('xray_image')
    temp_files = []

    # Validate inputs
    if not xray_image or xray_image.filename == '':
        logger.error("No image uploaded")
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
    if not allowed_file(xray_image.filename, current_app):
        logger.error("Invalid file type for image")
        return jsonify({'success': False, 'error': 'Invalid file type for image'}), 400

    # Save image
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(xray_image.filename))
    try:
        xray_image.save(filepath)
        temp_files.append(filepath)
        logger.info(f"Saved image: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save image: {str(e)}")
        return jsonify({'success': False, 'error': f'Failed to save image: {str(e)}'}), 500

    try:
        # Load and predict with the combined model
        model = load_model()
        processed = preprocess_image(filepath, target_size=(224, 224))
        pred = model.predict(processed, verbose=0)
        gender = 'Female' if pred[0][0] > 0.5 else 'Male'
        conf = round(pred[0][0] * 100 if gender == 'Female' else (1 - pred[0][0]) * 100, 1)
        result = {'gender': gender, 'confidence': conf}
        logger.info(f"Prediction: gender={gender}, confidence={conf}")

        # Unload model
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        logger.info("Combined model unloaded")

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        cleanup_files(temp_files)
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500

    finally:
        cleanup_files(temp_files)
        gc.collect()

    return jsonify({'success': True, 'result': result})
