import logging
import os
from flask import Flask
from routes import main
from auth import auth

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
logger.info("Flask app initialized")

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info(f"Uploads directory created/verified: {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    logger.error(f"Failed to create uploads directory: {str(e)}")
    raise

# Register Blueprints
app.register_blueprint(auth)
app.register_blueprint(main)
logger.info("Blueprints registered: auth, main")

# Health check for Render
@app.route("/health")
def health():
    logger.info("Health check accessed")
    return "Sex Estimator Classifier API is running", 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))  # Default to Render's port 10000
    host = '0.0.0.0'  # Bind to all interfaces for Render
    logger.info(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=False)  # Disable debug for production

