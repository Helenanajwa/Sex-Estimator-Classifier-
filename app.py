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

# Register Blueprints
app.register_blueprint(auth)
app.register_blueprint(main)

# Optional: Basic root route for health check
@app.route("/")
def home():
    return "Sex Estimator Classifier API is running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
