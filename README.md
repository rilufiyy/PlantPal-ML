# PlantPal CNN Model - README
- Sri Lutfiya Dwiyeni (M200B4KX4198)
- Siti Alya Nurrohmah (M200B4KX4164)
- Nadia Adyutarahma Putri (M200B4KX4164)

## PlantPal Dataset Access
https://www.kaggle.com/datasets/srilutfiyadwiyeni/plantpal-dataset

# Overview
This project demonstrates the implementation, training, and deployment of a Convolutional Neural Network (CNN) model for plant disease classification using TensorFlow. The model is capable of identifying 31 distinct classes of plant diseases based on input images, including healthy states of plants.

# Visualization of training metrics (accuracy and loss):
- train acc: 95.59% and val acc: 97.16%
- train loss: 0.2768 and val loss 0.2209

# Model serialization into multiple formats:
- Keras (.h5 and .keras)
- TensorFlow SavedModel
- TensorFlow Lite
- TensorFlow.js
- .json

# Requirements
The following Python packages are required for running the project:
- ipython==8.27.0
- ipywidgets==7.8.1
- kagglehub==0.3.4
- keras==2.15.0
- matplotlib==3.7.1
- numpy==2.2.0
- pandas==2.2.3
- Pillow==11.0.0
- plotly==5.24.1
- protobuf==4.25.3
- pydot==3.0.3
- scikit_learn==1.6.0
- seaborn==0.13.2
- tensorflow==2.15.0
- tensorflow_intel==2.15.0
- tensorflowjs==4.22.0

# Install the dependencies using pip:
pip install -r requirements.txt

# Usage

1. Data Preparation

Place the training and validation images into respective folders:

training_data/

validation_data/

Images should be organized into subdirectories for each class.


2. Training the Model

Run the script to train the CNN model:

python scripts/train_model.py

The model will be saved in multiple formats after training, and a history.pkl file will contain the training metrics.

3. Evaluating the Model

Use the evaluation script to generate metrics such as precision, recall, and F1-score:

python scripts/evaluate_model.py

4. Predicting on New Images

Run the prediction script to test the model on new images:

python scripts/predict_image.py

Upload an image using the interactive widget, and the script will display the top predictions with confidence scores.

5. Deployment

TensorFlow.js

Convert the model to TensorFlow.js format for web deployment:

tensorflowjs_converter --input_format=tf_saved_model saved_model/plantpal_cnn_model tfjs_cnn_model

TensorFlow Lite

Use the .tflite model for edge device deployment.

6. Download Trained Models

A zipped archive models_cnn.zip containing all serialized model formats can be downloaded.

# Visualization

Accuracy and Loss

The training and validation accuracy and loss curves are saved as images:

Accuracy_CNN.png

Loss_CNN.png

# Confusion Matrix

The confusion matrix visualization can be generated using the evaluation script.
<p align="center">
  <img src="https://github.com/user-attachments/assets/1bc3fdd6-64ce-417a-8d1f-34435d0c043a" alt="image" width="400"/>
</p>


# Notes

Ensure TensorFlow version 2.15.0 or compatible is installed for successful execution.

GPU acceleration is recommended for training.

Use the provided requirements.txt to ensure compatibility.

# License

This project is open-source and available under the MIT License.

# Acknowledgments

TensorFlow Documentation

Kaggle Plant Pathology datasets

For questions or issues, please open an issue in the repository.

