from flask import Flask
import os
from routes import main
from auth import auth

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

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
