import os
import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog

# Function to display the original and processed image along with prediction
def display_prediction(image_path):
    original_image = cv2.imread(image_path)
    processed_image = cv2.resize(original_image, (128, 128)) / 255.0
    
    freshness_level, confidence_percentage = predict_freshness(image_path)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(processed_image)
    ax[1].set_title('Processed Image (128x128)')
    ax[1].axis('off')
    
    plt.figtext(0.5, 0.01, f'Predicted Freshness Level: {freshness_level} (Confidence: {confidence_percentage:.2f}%)', 
                ha='center', fontsize=12, color='black')
    plt.show()

# Function to load images from the new structure
def load_images(image_folder):
    images = []
    labels = []
    for fruit_folder in os.listdir(image_folder):
        fruit_path = os.path.join(image_folder, fruit_folder)
        if os.path.isdir(fruit_path):
            for class_folder in os.listdir(fruit_path):
                class_path = os.path.join(fruit_path, class_folder)
                if os.path.isdir(class_path):
                    for image_path in glob.glob(os.path.join(class_path, '*.jpg')):  # Adjust the extension if needed
                        image = cv2.imread(image_path)
                        if image is not None:
                            image = cv2.resize(image, (128, 128))
                            images.append(image)
                            if class_folder == "Fresh":
                                labels.append(1)
                            elif class_folder == "Spoiled":
                                labels.append(0)
                        else:
                            print(f"Warning: Unable to read image {image_path}")
    print(f"Total images loaded: {len(images)}")
    print(f"Total labels loaded: {len(labels)}")
    return np.array(images), np.array(labels)

# Load your dataset
image_folder = r'C:\Users\iic08\OneDrive\Desktop\FLIPKART\USECASE 4\dataset'
X, y = load_images(image_folder)

# Normalize the images
X = X / 255.0

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

# Create generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

# Build a transfer learning model using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(train_generator, 
                    epochs=20, 
                    validation_data=test_generator)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Function to predict freshness level with confidence percentage
def predict_freshness(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    prediction = model.predict(np.array([image]))[0][0]
    freshness_level = "Fresh" if prediction > 0.5 else "Spoiled"
    confidence_percentage = prediction * 100 if freshness_level == "Fresh" else (1 - prediction) * 100
    return freshness_level, confidence_percentage

# Function to select an image file
def select_image_file():
    # Create a Tk root widget (hidden)
    root = Tk()
    root.withdraw()  # Hide the root window
    # Open the file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
    )
    return file_path

# Loop to allow multiple image predictions
while True:
    test_image_path = select_image_file()
    if test_image_path:  # Check if a file was selected
        result, confidence = predict_freshness(test_image_path)
        print(f'Predicted Freshness Level: {result} (Confidence: {confidence:.2f}%)')
        display_prediction(test_image_path)
        
        # Ask user if they want to scan another image
        another = input("Do you want to scan another image? (yes/no): ").strip().lower()
        if another != 'yes':
            break
    else:
        print("No file selected.")
        break
