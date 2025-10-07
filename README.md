# Customer-Churn-Prediction-using-ANN

# 📉 Telco Customer Churn Prediction using Artificial Neural Networks (ANN)

This project implements a complete Machine Learning pipeline to predict customer churn in a telecommunications company using a Feed-Forward Artificial Neural Network (ANN).
## 🚀 Project Overview

The goal is to classify customers as likely to 'Churn' (leave) or 'Not Churn' based on their service usage, contract details, and demographic information.

### Features :
* **Real-World Data:** Utilizes the industry-standard Telco Customer Churn dataset.
* **Data Preprocessing:** Implements a robust `ColumnTransformer` pipeline for handling mixed data types.
    * **Feature Engineering:** One-Hot Encoding for categorical features.
    * **Feature Scaling:** Standardization (`StandardScaler`) for numerical features.
* **ANN Architecture:** A Sequential Keras model with multiple Dense layers and a **Dropout** layer for regularization to prevent overfitting.

## 🛠️ Technology Stack

* **Language:** Python
* **Libraries:**
    * `TensorFlow / Keras`: For building and training the ANN.
    * `Scikit-learn`: For data splitting, preprocessing pipelines, and feature scaling.
    * `Pandas / NumPy`: For data manipulation.
    * `Requests / IO`: For dynamically loading the external dataset.

## ⚙️ Installation and Setup

To run this project, you need Python and the necessary libraries (TensorFlow / Keras,Scikit-learn,Requests / IO) installed .

###  Prerequisites

Ensure you have a modern version of Python installed.

###📊 Results
The model outputs the test accuracy, providing a concrete measure of its performance in predicting customer churn based on the prepared features. The inclusion of Dropout layers helps demonstrate techniques for building more robust production models.

