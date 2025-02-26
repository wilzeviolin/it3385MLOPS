import pickle
from flask import Flask, request, jsonify
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
def load_model():
    try:
        with open('seed_pipline.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            return model
    except pickle.UnpicklingError as e:
        print(f"Error unpickling the model: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Load the model once when the app starts
model = load_model()

@app.route('/')
def home():
    return "Wheat Type Classifier API is Running!"

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the request
    data = request.get_json(force=True)
    
    # Extract the features
    area = data['Area']
    perimeter = data['Perimeter']
    compactness = data['Compactness']
    length = data['Length']
    width = data['Width']
    asymmetry_coeff = data['AsymmetryCoeff']
    groove = data['Groove']
    
    # Prepare the input features as a numpy array for prediction
    features = np.array([[area, perimeter, compactness, length, width, asymmetry_coeff, groove]])
    
    # Make the prediction
    prediction = model.predict(features)
    
    # Return the prediction as a response
    return jsonify({"predicted_wheat_type": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
