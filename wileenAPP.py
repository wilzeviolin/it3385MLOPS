import pickle
import numpy as np

# Load the trained model
def load_model():
    try:
        with open('seed_type_classification.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            print("Model loaded successfully.")
            return model
    except pickle.UnpicklingError as e:
        print(f"Error unpickling the model: {e}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
    return None  # Return None if the model loading fails

# Load the model once when the app starts
model = load_model()

# Function to get input (modify this based on your needs)
def get_input():
    # You can either ask for real-time input (e.g., using `input()`) or use hardcoded test data
    print("Using the following sample input for prediction:")

    # Hardcoded input data for testing (can be replaced with actual input logic)
    area = 15.5
    perimeter = 14.3
    compactness = 0.8
    length = 7.1
    width = 3.5
    asymmetry_coeff = 2.1
    groove = 5.0

    print(f"Using input values: Area={area}, Perimeter={perimeter}, Length={length}, etc.")

    return np.array([[area, perimeter, compactness, length, width, asymmetry_coeff, groove]])

# Function to make prediction
def predict(features):
    if model is not None:
        prediction = model.predict(features)
        print(f"Predicted Wheat Type: {int(prediction[0])}")
    else:
        print("Model is not loaded. Cannot make predictions.")

if __name__ == '__main__':
    while True:
        features = get_input()  # Get features from user
        predict(features)       # Make prediction
        cont = input("Do you want to make another prediction? (y/n): ")
        if cont.lower() != 'y':
            break
