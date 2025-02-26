import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import os
from sklearn.linear_model import RidgeClassifier


# Load the trained model
def load_model():
    try:
        with open('seed_type_classification.pkl', 'wb') as f:
            pickle.dump(model, f)
            return model
    except pickle.UnpicklingError as e:
        print(f"Error unpickling the model: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Load the model once when the app starts
model = load_model()

# Function to get input from the user
def get_input():
    print("Using the following sample input for prediction:")

    # Hardcoded values for testing
    area = 15.5
    perimeter = 14.3
    compactness = 0.8
    length = 7.1
    width = 3.5
    asymmetry_coeff = 2.1
    groove = 5.0

    print(f"Area: {area}, Perimeter: {perimeter}, Length: {length}, etc.")

    return np.array([[area, perimeter, compactness, length, width, asymmetry_coeff, groove]])
    
# Function to make prediction
def predict(features):
    prediction = model.predict(features)
    print(f"Predicted Wheat Type: {int(prediction[0])}")

if __name__ == '__main__':
    while True:
        features = get_input()  # Get features from user
        predict(features)       # Make prediction
        cont = input("Do you want to make another prediction? (y/n): ")
        if cont.lower() != 'y':
            break
