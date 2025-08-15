ForensRadioAI
Description
ForensRadioAI is an AI-powered tool for skeletal sex estimation using X-ray images, primarily of the skull from AP (anteroposterior) and lateral views. Users can upload X-ray images, and the application employs a Convolutional Neural Network (CNN) to estimate the gender of the subject. The project leverages data analytics methods, including training and testing on image datasets, to achieve accurate sex estimation.
Table of Contents

Installation
Usage
Technologies Used
Contributing
License
Contact

Installation
To set up ForensRadioAI locally, ensure you have Python installed. Follow these steps to install the required dependencies:
# Clone the repository
git clone https://github.com/Helenanajwa/Sex-Estimator-Classifier-.git
cd Sex-Estimator-Classifier-

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Requirements
The following dependencies are listed in requirements.txt:

absl-py==2.3.1
astunparse==1.6.3
blinker==1.9.0
certifi==2025.7.14
charset-normalizer==3.4.2
click==8.2.1
colorama==0.4.6
Flask==3.1.1
flatbuffers==25.2.10
gast==0.6.0
google-pasta==0.2.0
grpcio==1.74.0
gunicorn==22.0.0
h5py==3.11.0
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
keras==2.14.0
libclang==18.1.1
Markdown==3.8.2
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
ml-dtypes==0.2.0
namex==0.1.0
numpy==1.23.5
opencv-python-headless==4.9.0.80
opt_einsum==3.4.0
optree==0.16.0
packaging==25.0
Pillow==11.3.0
protobuf==4.25.8
Pygments==2.19.0
requests==2.32.4
rich==14.0.0
setuptools>=40.8.0,<70.0.0
six==1.17.0
tensorboard==2.14.0
tensorboard-data-server==0.7.2
tensorflow-cpu==2.14.0
termcolor==3.1.0
typing_extensions==4.14.1
urllib3==2.5.0
Werkzeug==3.1.3
wheel==0.44.0
wrapt>=1.14.0,<1.15

Usage
To run the ForensRadioAI application locally:
# Start the Flask application with Gunicorn
gunicorn -w 4 --timeout 600 -b 0.0.0.0:10000 app:app


Access the application in your browser at http://localhost:10000.
Log in using the login page (see below).
Upload X-ray images (skull, AP, and lateral views) via the index.html or upload.html interface.
The application will process the images using the pre-trained CNN model (model.py) and display the estimated gender (see result example below).

Key Python Files

app.py: The main application file that initializes the Flask web server and integrates the project components.
auth.py: Handles user authentication and security features for the application.
routes.py: Defines the routes and endpoints for the web application, managing navigation and API calls (e.g., image upload and gender estimation).

Training the Model
To train the CNN model:

Prepare a dataset of labeled X-ray images (AP and lateral views) and place them in the appropriate directories (e.g., static or uploads).
Use the model.py script along with pre-trained models (best_model.h5, best_model_AP.h5, best_model_lateral.h5) to train or fine-tune the model.
Run training scripts in a Python environment (e.g., VS Code) using the following example command:

python model.py

Screenshots

Login Page:The login interface for accessing the application.

Result After Upload (Sample Image):Example result showing a female sex estimation with 80.2% probability using a sample AP view X-ray.


Deployment
The application is deployed on Render. Visit https://sex-estimator-classifier.onrender.com to access the live version.
Technologies Used

Python: Core programming language for model training and application logic.
TensorFlow/Keras: For building and training the Convolutional Neural Network (CNN).
Flask: Web framework for the application’s frontend and backend.
Gunicorn: WSGI server for running the Flask application.
OpenCV: For image processing (opencv-python-headless).
Pillow: For handling image uploads.
VS Code: Recommended IDE for development and training.
Render: Hosting platform for deployment.

Contributing
Contributions are welcome! To contribute to ForensRadioAI:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a pull request with a detailed description of your changes.

Please ensure your code follows the project’s style guidelines and includes relevant tests.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact

GitHub: @Helenanajwa
Email: s65569@ocean.umt.edu.my
