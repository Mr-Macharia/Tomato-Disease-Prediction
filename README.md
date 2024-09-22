## Leaf Disease Classification Model
The Dataset was from Kaggle -> https://www.kaggle.com/datasets/teresha/tomato-disease
This project implements a Convolutional Neural Network (CNN) to classify leaf diseases using a dataset of 60,387 images across 10 classes. The goal is to develop a robust model to predict leaf diseases accurately, which will be later integrated into an app for real-time disease detection.


### Overview
This repository contains the implementation of a deep learning model trained to classify leaf diseases from images. The model is trained using TensorFlow and Keras, utilizing a CNN architecture optimized for image classification tasks.

### Dataset
The dataset contains **60,387 images** categorized into **10 classes** of leaf diseases. The images were preprocessed and split into training and validation sets to develop and evaluate the model.

### Model Architecture
The model is built using a Convolutional Neural Network (CNN) with the following layers:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification

The model is trained using the **Adam optimizer** and **categorical cross-entropy loss function**. Regularization techniques such as **dropout** were used to prevent overfitting.

### Training Process
The model was trained using:
- **Batch size**: 32
- **Epochs**: 50
- **Optimizer**: Adam
- **Loss function**: Categorical Crossentropy

Training and validation graphs indicate smooth convergence with no signs of overfitting, validating the effectiveness of the architecture and regularization techniques used.

### Model Performance
The trained model achieved the following performance metrics:

- **Accuracy**: Approximately 97.9% on the validation dataset
- **Precision**: 0.9790
- **Recall**: 0.9641
- **Training Loss**: The training loss steadily decreased during the training process, showing strong learning.
- **Validation Loss**: The validation loss remained stable, indicating that the model generalizes well to unseen data.

### Next Steps
The next phase of the project is to integrate this trained model into a mobile application to enable real-time leaf disease detection. The app will allow users to upload images of leaves and receive predictions regarding potential diseases.
