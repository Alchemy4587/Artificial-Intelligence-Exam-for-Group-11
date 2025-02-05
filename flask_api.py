import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'models/fashion_mnist_model.h5'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Convert to grayscale and resize
    image = image.convert('L')
    image = image.resize((28, 28))
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    # Reshape for model
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Read and preprocess image
        image = Image.open(BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        
        result = {
            'category': class_names[class_index],
            'confidence': confidence * 100
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)