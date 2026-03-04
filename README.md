# AI Malaria Detection System 🔬🩸

This is a comprehensive, end-to-end Deep Learning project for detecting malaria parasites in thin blood smear microscopic images. It uses a **MobileNetV2** Convolutional Neural Network (CNN) architecture via Transfer Learning to classify images as either **Parasitized** or **Uninfected**.

The project features a full Python training pipeline (compatible with Kaggle datasets) and a professional Medical Dashboard built with Flask and Bootstrap 5 to seamlessly serve the AI model's predictions.

---

## Features ✨

### 🏗️ Deep Learning Training Pipeline

- **Automated Dataset Handling**: Dynamically extracts the Kaggle Cell Images dataset (`parasitized.zip` and `uninfected.zip`).
- **Data Augmentation**: Robust preprocessing loops using Keras `ImageDataGenerator` (rotation, zooming, horizontal flips).
- **Transfer Learning**: Built on MobileNetV2 with pre-trained ImageNet weights, fine-tuned with custom classifier layers.
- **Integrated Evaluation**: Automatically plots training/validation accuracy, generates a Classification Report, and exports a Seaborn Confusion Matrix (`confusion_matrix.png`).

### 🏥 Medical Dashboard Web App

- **Modern UI**: A clean, responsive, and professional dashboard interface matching modern healthcare analytics platforms, built with HTML5, CSS3, and Bootstrap 5.
- **Instant Predictions**: Drag-and-drop file upload with a dynamic loading spinner simulating an AI analysis process.
- **Comprehensive Results**: Displays clear classification badges, an animated confidence score meter, and medical action recommendations without requiring page reloads.

---

## Tech Stack 🛠️

- **Backend core**: Python 3.11, Flask
- **Frontend**: HTML5, custom CSS, Bootstrap 5, FontAwesome, JavaScript (AJAX)
- **Deep Learning**: TensorFlow / Keras (MobileNetV2)
- **Data Science**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## Project Structure 📂

```text
Malaria-Detection/
│
├── app.py                 # Main Flask application serving the web dashboard
├── train.py               # Complete Kaggle training, augmentation & evaluation pipeline
├── requirements.txt       # Project dependencies
├── static/                
│   └── style.css          # Custom medical UI styling
├── templates/             
│   ├── base.html          # Base layout template with sidebar & navbar
│   └── dashboard.html     # Main interactive prediction interface
└── ...
```

---

## Installation & Setup 🚀

> **Note:** TensorFlow currently requires **Python 3.11** (or 3.10). It is **not** supported on Python 3.14. Please ensure you are running a supported Python version.

### 1. Clone the repository

```bash
git clone https://github.com/raghuvanshi-sec/Malaria-Detection.git
cd Malaria-Detection
```

### 2. Set up the Environment

It is highly recommended to use a virtual environment configured with python 3.11:

```bash
# Using Windows PowerShell
& "C:\Program Files\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Training Pipeline (Optional)

If you wish to train the model from scratch, ensure you have the Kaggle Maleria Cell Images dataset (`parasitized.zip` and `uninfected.zip`) in the root directory.

```bash
python train.py

```

This will extract the images, train the MobileNetV2 model for 5-10 epochs, output evaluation metrics, and generate `malaria_model.h5`.

### 4. Run the Medical Dashboard

Start the Flask web server to interact with your trained model:

```bash
python app.py
```

Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Disclaimer ⚠️

This tool is for educational, research, and hackathon demonstration purposes only. It is **not** intended to be a substitute for professional medical advice, diagnosis, or clinical treatment workflows. Always seek the advice of a qualified health provider regarding a medical condition.
