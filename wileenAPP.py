import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import os

# Load the trained model
def load_model():
    try:
        with open('seed_type_classification.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            return model
    except pickle.UnpicklingError as e:
        print(f"Error unpickling the model: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Load the model once when the app starts
model = load_model()

# Function to get input from the user
def get_input():
    print("Enter the following features to predict the wheat type:")

    area = float(input("Area: "))
    perimeter = float(input("Perimeter: "))
    compactness = float(input("Compactness: "))
    length = float(input("Length: "))
    width = float(input("Width: "))
    asymmetry_coeff = float(input("Asymmetry Coefficient: "))
    groove = float(input("Groove: "))

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
