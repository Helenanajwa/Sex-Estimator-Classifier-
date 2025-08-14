from flask import Flask
from routes import main
from auth import auth
import os
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Limit uploads to 1MB

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    logger.info(f"Uploads directory created: {app.config['UPLOAD_FOLDER']}")

# Register blueprints
app.register_blueprint(main)
app.register_blueprint(auth)
logger.info("Blueprints registered successfully")

# Log registered routes
logger.info(f"Registered routes: {[rule.rule for rule in app.url_map.iter_rules()]}")

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production

