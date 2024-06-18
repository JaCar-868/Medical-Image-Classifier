# AI_Engineering_Capstone_Project.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import torch

# Set seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Load the data
def load_data(directory, positive_dir, negative_dir):
    positive_files = [os.path.join(directory, positive_dir, file) for file in os.listdir(os.path.join(directory, positive_dir)) if file.endswith(".pt")]
    negative_files = [os.path.join(directory, negative_dir, file) for file in os.listdir(os.path.join(directory, negative_dir)) if file.endswith(".pt")]

    all_files = positive_files + negative_files
    labels = np.array([1] * len(positive_files) + [0] * len(negative_files))

    data = []
    for file in all_files:
        image = torch.load(file).numpy()  # Load the tensor and convert to numpy array
        image = np.transpose(image, (1, 2, 0))  # Adjust dimensions if necessary
        data.append(image)
    data = np.array(data)

    return data, labels

# Directories and file paths
directory = "/home/wsuser/work"
positive_dir = "Positive_tensors"
negative_dir = "Negative_tensors"

# Load training and validation data
data, labels = load_data(directory, positive_dir, negative_dir)

# Split into training and validation sets
train_data, val_data = data[:30000], data[30000:]
train_labels, val_labels = labels[:30000], labels[30000:]

# Preprocess the data
train_data = train_data / 255.0
val_data = val_data / 255.0

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(train_data)

# Visualize some sample images
def plot_sample_images(data, labels, title, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(data[i])
        plt.title(f"{title} - {labels[i]}")
        plt.axis('off')
    plt.show()

# Plot sample training images
plot_sample_images(train_data, train_labels, "Training Image")

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Add dropout for regularization
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(datagen.flow(train_data, train_labels, batch_size=100),
                    epochs=50,
                    validation_data=(val_data, val_labels),
                    callbacks=[early_stopping, model_checkpoint])

# Plot training loss and validation accuracy
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# Load the best model
model.load_weights('best_model.h5')

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_data, val_labels)
print(f'Validation accuracy: {val_accuracy:.2f}')

# Plot some misclassified samples
predictions = model.predict(val_data)
predicted_labels = np.argmax(predictions, axis=1)

count = 0
for i in range(len(val_labels)):
    if predicted_labels[i] != val_labels[i] and count < 4:
        plt.imshow(val_data[i])
        plt.title(f"Predicted: {predicted_labels[i]}, Actual: {val_labels[i]}")
        plt.show()
        count += 1
