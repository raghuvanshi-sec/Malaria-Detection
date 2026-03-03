import os
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report, confusion_matrix

def extract_dataset():
    """
    Extracts the dataset ZIP files into the appropriate directory structure.
    Expects 'parasitized.zip' and 'uninfected.zip' in the current directory.
    """
    dataset_dir = 'cell_images'
    parasitized_dir = os.path.join(dataset_dir, 'Parasitized')
    uninfected_dir = os.path.join(dataset_dir, 'Uninfected')
    
    # Create the root dataset folder if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    # Extract Parasitized
    if not os.path.exists(parasitized_dir):
        if os.path.exists('parasitized.zip'):
            print("Extracting parasitized.zip...")
            with zipfile.ZipFile('parasitized.zip', 'r') as zip_ref:
                # Assuming the zip extracts the files directly or into a single folder
                zip_ref.extractall(parasitized_dir)
            print("Successfully extracted Parasitized dataset.")
        else:
            print("Warning: 'parasitized.zip' not found. Ensure dataset is present or manually placed in cell_images/Parasitized/")
            
    # Extract Uninfected
    if not os.path.exists(uninfected_dir):
        if os.path.exists('uninfected.zip'):
            print("Extracting uninfected.zip...")
            with zipfile.ZipFile('uninfected.zip', 'r') as zip_ref:
                zip_ref.extractall(uninfected_dir)
            print("Successfully extracted Uninfected dataset.")
        else:
            print("Warning: 'uninfected.zip' not found. Ensure dataset is present or manually placed in cell_images/Uninfected/")
            
    return dataset_dir

def build_model(input_shape=(128, 128, 3)):
    """
    Build the MobileNetV2 transfer learning model for binary classification.
    """
    print("Building MobileNetV2 Transfer Learning Model...")
    # Load the base MobileNetV2 model with ImageNet weights, excluding the top classifier
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classifier layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Combine the base model and custom classifier
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    return model

def plot_history(history):
    """
    Output training and validation accuracy/loss plots.
    """
    print("\nPlotting Training & Validation Metrics...")
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_accuracy.png')
    print("Metrics plot saved to 'training_accuracy.png'.")
    plt.close()

def evaluate_model(model, validation_generator):
    """
    Generate predictions, classification report, and seaborn confusion matrix.
    """
    print("\n--- Starting Model Evaluation ---")
    
    # Reset generator before prediction to ensure order matches classes exactly
    validation_generator.reset()
    
    # Predictions will be probabilities between 0 and 1
    predictions = model.predict(validation_generator, steps=len(validation_generator))
    
    # Convert probabilities to binary classes (0 or 1) using a 0.5 threshold
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels
    y_true = validation_generator.classes
    
    # Get class names [0: Parasitized, 1: Uninfected]
    class_names = list(validation_generator.class_indices.keys())
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting Confusion Matrix with Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Malaria Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the confusion matrix to a file
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'.")
    plt.close()

def main():
    # 1. Prepare Dataset
    dataset_dir = extract_dataset()
    
    # Check if subdirectories actually have files
    parasitized_dir = os.path.join(dataset_dir, 'Parasitized')
    uninfected_dir = os.path.join(dataset_dir, 'Uninfected')
    
    if not os.path.exists(parasitized_dir) or not os.path.exists(uninfected_dir):
         print(f"Error: Dataset directories not ready. Ensure ZIPs were extracted successfully.")
         return
         
    p_count = len(os.listdir(parasitized_dir)) if os.path.exists(parasitized_dir) else 0
    u_count = len(os.listdir(uninfected_dir)) if os.path.exists(uninfected_dir) else 0
    
    print(f"Current Dataset Status: Parasitized={p_count} images, Uninfected={u_count} images.")
    if p_count == 0 or u_count == 0:
        print("Error: One or both datasets are empty. Please check the 'cell_images' folder structure.")
        return
    
    # Image dimensions and training parameters
    img_height, img_width = 128, 128
    batch_size = 32
    epochs = 10 # 5-10 Epochs as requested
    
    print("\nInitializing Data Generators with Augmentation...")
    
    # 2. Image Data Generator configuration
    # Training Data Generator (rescaling + data augmentation + 80/20 split)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
        validation_split=0.2
    )
    
    # Validation Data Generator (ONLY rescaling + 80/20 split)
    # Never apply augmentations to the validation test set!
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # 3. Load Data 
    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True # Shuffle during training
    )
    
    print("Loading Validation Data...")
    validation_generator = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False # DO NOT shuffle validation data (important for confusion matrix)
    )

    # 4. Build and Train Model
    model = build_model(input_shape=(img_height, img_width, 3))
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nStarting Training Phase...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    
    # 5. Saving the Model
    print("\nSaving Model...")
    model.save('malaria_model.h5')
    print("Model successfully saved to 'malaria_model.h5'.")
    
    # 6. Evaluation and Visualization
    plot_history(history)
    evaluate_model(model, validation_generator)
    
    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == '__main__':
    main()
