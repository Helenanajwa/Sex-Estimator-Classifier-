from flask import Flask
import os
from routes import main
from auth import auth

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Config
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Routes
app.register_blueprint(auth)
app.register_blueprint(main)

# Optional root route (for Render's health check)
@app.route('/')
def home():
    return 'Sex Estimator Classifier API is running'

# Entry point (used for local development only)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render provides PORT env
    app.run(host='0.0.0.0', port=port, debug=True)


