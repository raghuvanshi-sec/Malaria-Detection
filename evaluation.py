import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

def main():
    model_path = 'malaria_model.h5'
    dataset_dir = 'cell_images'
    
    # Image dimensions
    img_height, img_width = 128, 128
    batch_size = 32
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
        return
        
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return

    print("Loading Model...")
    model = load_model(model_path)
    
    print("Initializing Data Generator for Validation Set...")
    # Rescale and split
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    validation_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False  # Important for keeping order for confusion matrix
    )
    
    if validation_generator.samples == 0:
        print("Error: No validation images found. Ensure the dataset directories are correctly populated.")
        return
        
    print("Evaluating Model and Generating Predictions...")
    # Predictions will be probabilities between 0 and 1
    predictions = model.predict(validation_generator)
    
    # Convert probabilities to binary classes (0 or 1) using a 0.5 threshold
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels
    y_true = validation_generator.classes
    
    # Get class names
    class_names = list(validation_generator.class_indices.keys())
    
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting Confusion Matrix with Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the confusion matrix to a file
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'.")
    plt.show()

if __name__ == '__main__':
    main()
