# Digit Recognizer using Fully Connected Network

This project implements a digit recognizer using a **Fully Connected Neural Network** (Multi-Layer Perceptron) on the **MNIST** dataset from Kaggle.

## Dataset
- Source: [MNIST Dataset on Kaggle](https://www.kaggle.com/c/digit-recognizer)
- The dataset contains 70,000 images of handwritten digits (0–9), each of size **28x28** pixels in grayscale.
- Training set: 60,000 images  
- Test set: 10,000 images

## 🧠 Model Architecture
Input image (28x28 = 784 features) --> [Affine layer - (Batch normalization) - ReLU activate - (Drop out)] x N --> Affine layer --> Softmax classification
(): Optional
This repo using architecture with N equal to 2
#1 hidden layer: 500 neurons
#2 hidden layer: 100 neurons

## ⚙Update Rules Comparison
This project includes an experiment comparing different optimization update rules:
1. **SGD** – Stochastic Gradient Descent
2. **SGD with Momentum**
3. **RMSProp**
4. **Adam**

Each optimizer is tested with the same model architecture, hyperparameters and small dataset from full dataset to evaluate:
- Convergence speed
- Final accuracy
- Training loss curve

# Results
<img width="699" height="720" alt="Results" src="https://github.com/user-attachments/assets/9d8386a8-c163-4af6-978e-fde0533494c9" />

## Training
First, tuning the hyperparameters (learning rate, weight scale, regularization hyperparameter) using random search from large range to smaller range
Traing with best hyperparamters, with Adam update rule, Batch normaliztion, dropout rate = 0.25

## Results
Validation accuracy: 0.944 (= 94.4%)
Test accuracy (on Kaggle): 0.964 (= 96.4%)
