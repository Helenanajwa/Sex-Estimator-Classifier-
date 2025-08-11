import logging
import os
from flask import Flask
from routes import main
from auth import auth

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.info(f"Uploads directory created: {app.config['UPLOAD_FOLDER']}")

# Log current working directory and contents
logging.info(f"Current working directory: {os.getcwd()}")
logging.info(f"Directory contents: {os.listdir('.')}")

# Register Blueprints
try:
    app.register_blueprint(auth)
    app.register_blueprint(main)
    logging.info("Blueprints registered successfully")
except Exception as e:
    logging.error(f"Failed to register blueprints: {str(e)}")
    raise

# Basic root route for health check
@app.route("/")
def home():
    logging.info("Root route accessed")
    return "Sex Estimator Classifier API is running"

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))  # Use Render's default port 10000
    logging.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)
