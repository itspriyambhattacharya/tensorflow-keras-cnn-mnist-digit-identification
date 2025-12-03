# MNIST Handwritten Digit Recognition using TensorFlow/Keras (CNN Model)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-important)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

This repository contains a complete implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The project demonstrates data preprocessing, model construction, training, evaluation, and prediction.

## Project Overview

Handwritten digit recognition serves as the basis for several OCR systems. This project implements a deep learning model capable of recognizing digits (0–9) from 28×28 grayscale images.

The CNN architecture includes:

- Convolutional layers with ReLU activation
- Max-Pooling layers
- Dense layers
- Softmax classifier

## Technologies Used

| Component | Technology         |
| --------- | ------------------ |
| Language  | Python             |
| Framework | TensorFlow / Keras |
| Dataset   | MNIST              |
| Model     | CNN                |

## Dataset Information – MNIST

- 60,000 training images
- 10,000 testing images
- Grayscale, 28×28
- Auto-loaded from Keras

## Data Preprocessing

### 1. Reshaping

Images are reshaped to add the channel dimension required by CNNs.

### 2. Normalization

Pixel values are scaled from (0–255) to (0–1).

## Model Architecture

The CNN model is built using the Keras Sequential API and includes:

- 3 Convolutional Layers
- 3 MaxPooling Layers
- Flatten Layer
- Dense Layer (64 neurons)
- Output Layer (10 neurons, softmax)

## Model Compilation

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metric: Accuracy

## Model Training

The model is trained for **6 epochs** with a batch size of **35**.

## Model Evaluation

Model accuracy is evaluated using the test dataset.

## Prediction

A single digit from the test set is used for prediction using `model.predict()`.

## How to Run the Notebook

### 1. Clone the Repository

```
git clone https://github.com/itspriyambhattacharya/tensorflow-keras-cnn-mnist-digit-identification.git
```

### 2. Install Dependencies

```
pip install tensorflow numpy
```

### 3. Run the Jupyter Notebook

Open the `.ipynb` file:

```
jupyter notebook number_recognition.ipynb
```

Or open it using VS Code:

```
code number_recognition.ipynb
```

## License

This project is licensed under the MIT License.

## Repository Link

https://github.com/itspriyambhattacharya/tensorflow-keras-cnn-mnist-digit-identification.git
