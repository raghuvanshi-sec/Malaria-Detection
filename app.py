from flask import Flask, render_template, request, jsonify
import os
import io
import time
import numpy as np
from PIL import Image
import hashlib
import random
from datetime import datetime, timedelta
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

from flask import send_from_directory
import os

@app.route('/cell_images/<path:filename>')
def serve_cell_images(filename):
    return send_from_directory('cell_images', filename)

@app.route('/', methods=['GET'])
def index():
    recent_patients = []
    
    # Helper to generate consistent mock data from filename
    def generate_mock_patient(filename, label):
        seed = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (10**8)
        rng = random.Random(seed)
        
        first_names = ["Jonathan", "Sarah", "Michael", "Emily", "David", "Jessica", "James", "Maria", "Robert", "Linda", "William", "Elizabeth"]
        last_names = ["Doe", "Jenkins", "Wong", "Rosa", "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia"]
        
        name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        age = rng.randint(5, 85)
        gender = rng.choice(['M', 'F'])
        pid = f"#PID-{rng.randint(1000, 9999)}"
        
        if label == "Parasitized":
            diagnosis = rng.choice(["Parasitized (P. falciparum)", "Parasitized (P. vivax)", "Parasitized (P. ovale)"])
            risk = "High Risk"
            confidence = round(rng.uniform(92.0, 99.8), 1)
            notes = f"Patient presented with high fever ({round(rng.uniform(38.5, 40.5), 1)}°C), chills, and fatigue. Travel history to endemic region noted. Immediate blood smear requested. AI flagged as High Risk ({diagnosis.split('(')[-1].strip(')')})."
            image_path = f"/cell_images/Parasitized/{filename}"
        else:
            diagnosis = "Uninfected"
            risk = "Low Risk"
            confidence = round(rng.uniform(94.0, 99.9), 1)
            notes = "Patient presented with mild symptoms. Routine blood screening conducted. Artificial Intelligence analysis indicates negative for malaria parasites."
            image_path = f"/cell_images/Uninfected/{filename}"
            
        days_ago = rng.randint(0, 5)
        date_str = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        return {
            'date': date_str,
            'id': pid,
            'name': name,
            'age': age,
            'gender': gender,
            'diagnosis': diagnosis,
            'confidence': confidence,
            'risk': risk,
            'filename': filename,
            'notes': notes,
            'image_path': image_path
        }


    parasitized_dir = os.path.join('cell_images', 'Parasitized')
    uninfected_dir = os.path.join('cell_images', 'Uninfected')
    
    try:
        if os.path.exists(parasitized_dir) and os.path.exists(uninfected_dir):
            p_files = [f for f in os.listdir(parasitized_dir) if f.endswith('.png')]
            u_files = [f for f in os.listdir(uninfected_dir) if f.endswith('.png')]
            
            p_sample = sorted(p_files)[:5]
            u_sample = sorted(u_files)[:5]
            
            for f in p_sample:
                recent_patients.append(generate_mock_patient(f, 'Parasitized'))
            for f in u_sample:
                recent_patients.append(generate_mock_patient(f, 'Uninfected'))
                
            if len(u_files) > 5:
                inc = generate_mock_patient(sorted(u_files)[5], 'Uninfected')
                inc['diagnosis'] = "Inconclusive (Re-test)"
                inc['risk'] = "Moderate"
                inc['confidence'] = round(random.uniform(55.0, 75.0), 1)
                recent_patients.append(inc)
                
            recent_patients.sort(key=lambda x: x['date'], reverse=True)
    except Exception as e:
        print(f"Failed parsing dataset: {e}")

    return render_template('dashboard.html', recent_patients=recent_patients)

@app.route('/api/realtime', methods=['GET'])
def realtime_data():
    import random
    import time
    
    # Simulate realistic fluctuations for dashboard metrics
    return jsonify({
        # Dashboard primary stats
        'patients_analyzed': 1204 + random.randint(0, 15),
        'high_risk_30d': 87 + random.randint(0, 3),
        'avg_confidence': round(random.uniform(97.5, 99.2), 1),
        
        # System Health (Settings Tab)
        'cpu_load': round(random.uniform(25.0, 45.0), 1),
        'gpu_temp': random.randint(65, 75),
        'memory_usage': round(random.uniform(4.0, 5.5), 1),
        'disk_io': round(random.uniform(10.0, 50.0), 1),
        'api_latency': random.randint(80, 150),
        'uptime_hours': 342,
        
        # Analytics Tab
        'accuracy': round(random.uniform(95.0, 96.5), 1),
        'precision': round(random.uniform(95.5, 96.8), 1),
        'recall': round(random.uniform(94.5, 96.0), 1),
        'f1_score': round(random.uniform(95.0, 96.5), 1),
        
        # Production Monitor
        'live_inference': random.randint(105, 125),
        'live_latency': random.randint(35, 60),
        
        'timestamp': time.time()
    }), 200

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

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://code.jquery.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com https://cdnjs.cloudflare.com; font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; img-src 'self' data: blob:"
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
