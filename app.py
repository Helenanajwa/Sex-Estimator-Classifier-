from flask import Flask
from routes import main, load_models
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    logger.info(f"Uploads directory created: {app.config['UPLOAD_FOLDER']}")

# Register blueprints
app.register_blueprint(main)
logger.info("Blueprints registered successfully")

# Log registered routes
logger.info(f"Registered routes: {[rule.rule for rule in app.url_map.iter_rules()]}")

# Load TensorFlow models at startup
try:
    load_models()
    logger.info("Models loaded successfully during app initialization")
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production
