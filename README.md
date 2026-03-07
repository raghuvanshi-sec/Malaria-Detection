# AI Malaria Detection System 🔬🩸

This is a comprehensive, end-to-end Deep Learning project for detecting malaria parasites in thin blood smear microscopic images. It uses a **MobileNetV2** Convolutional Neural Network (CNN) architecture via Transfer Learning to classify images as either **Parasitized** or **Uninfected**.

The project features a full Python training pipeline (compatible with Kaggle datasets) and a premium, dark-themed Medical Dashboard built with Flask and Bootstrap 5 to serve the AI model's predictions in real-time.

---

## Features ✨

### 🏗️ Deep Learning Training Pipeline

- **Automated Dataset Handling**: Dynamically extracts the Kaggle Cell Images dataset (`parasitized.zip` and `uninfected.zip`).
- **Data Augmentation**: Robust preprocessing loops using Keras `ImageDataGenerator` (rotation, zooming, horizontal flips).
- **Transfer Learning**: Built on MobileNetV2 with pre-trained ImageNet weights, fine-tuned with custom classifier layers.
- **Integrated Evaluation**: Automatically plots training/validation metrics and exports a Seaborn Confusion Matrix (`confusion_matrix.png`).

### 🏥 Premium Medical Dashboard

- **Next-Gen Dark UI**: A stunning, premium dark-mode interface featuring CSS glassmorphism, FontAwesome 6 icons, and smooth transitions.
- **Microscopic Analysis Module**: Drag-and-drop image upload area with real-time preview and dynamic loading states.
- **Diagnostic Reports**: Instant generation of diagnostic badges (Infected/Uninfected) with animated confidence bars and detailed analysis metadata.
- **Responsive Layout**: Optimized for both high-resolution medical monitors and mobile devices.

---

## Tech Stack 🛠️

- **Backend**: Python 3.11, Flask
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), Bootstrap 5, FontAwesome 6, JavaScript (AJAX)
- **Deep Learning**: TensorFlow 2.x / Keras (MobileNetV2)
- **Data Prep**: NumPy, PIL, Scikit-learn, Matplotlib, Seaborn

---

## Project Structure 📂

```text
Malaria-Detection/
│
├── app.py                 # Main Flask server with real-time prediction routes
├── train.py               # ML Pipeline: Augmentation, Training, & Evaluation
├── evaluation.py          # Post-training evaluation scripts
├── malaria_model.h5       # The trained MobileNetV2 weight file
├── static/                
│   └── style.css          # Premium dark-theme & glassmorphism styles
├── templates/             
│   ├── base.html          # Global layout & medical navigation
│   └── dashboard.html     # Interactive diagnostic control center
└── cell_images/           # Dataset directory (Parasitized/Uninfected)
```

---

## Installation & Setup 🚀

> [!IMPORTANT]
> TensorFlow currently requires **Python 3.11** for stable compatibility. It is **not** supported on Python 3.14.

### 1. Clone the repository

```bash
git clone https://github.com/raghuvanshi-sec/Malaria-Detection.git
cd Malaria-Detection
```

### 2. Set up the Environment

Create a virtual environment using Python 3.11 to ensure dependency compatibility:

```bash
# Windows
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Install required packages
pip install tensorflow matplotlib seaborn scikit-learn flask pillow
```

### 3. Run the Training Pipeline (Optional)

If you need to retrain the model, ensure the `cell_images` directory is populated:

```bash
python train.py
```

This will train for several epochs, save the updated `malaria_model.h5`, and generate a confusion matrix.

### 4. Launch the Dashboard

Start the medical server:

```bash
python app.py
```

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to begin clinical sample analysis.

---

## Disclaimer ⚠️

This tool is a screening demonstration for healthcare professionals and researchers. It is **not** a substitute for professional medical diagnosis or clinical verification. Always consult qualified medical personnel for patient treatment.
