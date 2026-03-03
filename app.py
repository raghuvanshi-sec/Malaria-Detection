from flask import Flask, render_template, request, jsonify
import os
import io
import time
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit uploads to 16MB max.

# Try to load the model at startup
MODEL_PATH = 'malaria_model.h5'
model = None
if os.path.exists(MODEL_PATH):
    print(f"Loading model {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Warning: Failed to load model. Ensure it is trained correctly. Error: {e}")
else:
    print(f"Warning: Model file {MODEL_PATH} not found. Please train the model first.")

# Target image size
IMG_HEIGHT, IMG_WIDTH = 128, 128

def process_image(img_file):
    """
    Open image, resize to target size, convert to array and scale.
    """
    img = Image.open(img_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img)
    
    # Rescale 1./255
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('dashboard.html')

@app.route('/records', methods=['GET'])
def records():
    return render_template('records.html')

@app.route('/analytics', methods=['GET'])
def analytics():
    return render_template('analytics.html')

@app.route('/settings', methods=['GET'])
def settings():
    return render_template('settings.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': "Model not loaded. Please train the model (train.py) and restart the server."}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': "No file part in the request."}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': "No selected file."}), 400
        
    try:
        # Preprocess the image
        img_array = process_image(file)
        
        # Simulate slight delay for progressive UI loaders (Optional)
        time.sleep(0.5) 
        
        # Predict
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        
        if score > 0.5:
            predicted_class = "Uninfected"
            confidence = score * 100
        else:
            predicted_class = "Parasitized"
            confidence = (1.0 - score) * 100
            
        result = {
            'class': predicted_class,
            'confidence': confidence,
            'success': True
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
