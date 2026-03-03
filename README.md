# Malaria Detection System 🔬🩸

This is an AI-powered medical dashboard web application for Malaria Detection. It allows users to upload microscopic images of blood smears, processes them through a deep learning model, and provides a prediction on whether the cell is Parasitized or Uninfected, along with a confidence score.

## Features ✨

- **Modern Medical Dashboard**: A clean, professional, and responsive UI built with HTML5, CSS3, and Bootstrap 5.
- **Deep Learning Model**: Uses a trained Convolutional Neural Network (CNN) for image classification.
- **Instant Predictions**: Seamless functional flow from image upload to prediction display.
- **Comprehensive Results**: Displays results, confidence scores, and medical recommendations.

## Tech Stack 🛠️

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Bootstrap 5, Jinja2 Templates
- **Machine Learning**: TensorFlow / Keras, NumPy, Pillow, OpenCV

## Project Structure 📂

Malaria-Detection/
│
├── app.py                 # Main Flask application entry point
├── train.py               # Script to train the deep learning model
├── evaluation.py          # Script for model evaluation
├── requirements.txt       # Python dependencies
├── static/                # Static assets (CSS, JS, Images, Uploads)
├── templates/             # HTML templates (index.html, dashboard.html, base.html)
└── ...

## Installation & Setup 🚀

1. **Clone the repository**:

   ```bash
   git clone https://github.com/raghuvanshi-sec/Malaria-Detection.git
   cd Malaria-Detection
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On Linux/macOS: source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:

   ```bash
   python app.py
   ```

5. **Access the Web App**:
   Open your browser and navigate to `http://127.0.0.1:5000`

## Disclaimer ⚠️

This tool is for educational and research purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
