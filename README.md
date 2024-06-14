# Medical-Image-Classifier using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) to analyze medical images for detecting and diagnosing diseases. The CNN model is built using TensorFlow's Keras API and trained on a dataset of medical images.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Medical image analysis using deep learning techniques can significantly aid in the detection and diagnosis of diseases. This project demonstrates the use of a CNN to classify medical images, helping healthcare professionals in making informed decisions.

## Installation

To run the code in this repository, you'll need Python installed along with the following libraries:
- TensorFlow
- NumPy

You can install the required libraries using pip:

pip install tensorflow numpy

## Usage
1. Clone the repository:

git clone https://github.com/yourusername/medical-image-analysis-cnn.git

cd medical-image-analysis-cnn

2. Prepare your dataset:

Organize your training and validation images in data/train and data/validation directories, respectively, with subdirectories for each class (e.g., data/train/class1, data/train/class2).

3. Run the training script:

python medical_image_analysis.py
The trained model will be saved as medical_image_analysis_model.h5.

## Model Architecture
The CNN model consists of three convolutional layers with ReLU activation and max pooling, followed by a flattening layer, a fully connected dense layer with dropout for regularization, and an output layer with sigmoid activation for binary classification.

## Training
The model is trained using the Adam optimizer and binary cross-entropy loss function. Early stopping is applied to prevent overfitting, monitoring the validation loss with a patience of 5 epochs.

## Contributing
Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue to discuss them.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/JaCar-868/Disease-Progression/blob/main/LICENSE) file for more details.
