import pickle
import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import traceback

# Initialize Flask app
wileen_app = Flask(__name__, template_folder='../templates')

# Load the trained model
def load_model():
    try:
        # Get the absolute path of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, 'artifacts', 'seed_pipeline.pkl')

        print(f"Attempting to load wheat model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Wheat model file not found at {model_path}")

        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
            print("Wheat model loaded successfully")
            return loaded_model

    except Exception as e:
        print(f"Error loading wheat model: {e}")
        print(traceback.format_exc())
        return None

# Load model at startup
model = load_model()

# Home route for rendering the wheat page
@wileen_app.route('/')
def home_page():
    return render_template('wheat.html')

# Processing route for predictions
@wileen_app.route('/process', methods=['POST'])
def process_form():
    print("Wheat process_form called")

    global model
    if model is None:
        print("Model not loaded")
        return jsonify({"error": "Model not loaded. Ensure 'seed_pipeline.pkl' exists in 'artifacts/' folder."})

    try:
        print(f"Received form data: {request.form}")

        # Extract and convert input data
        area = float(request.form['area'])
        perimeter = float(request.form['perimeter'])
        compactness = float(request.form['compactness'])
        length = float(request.form['length'])
        width = float(request.form['width'])
        asymmetry_coeff = float(request.form['asymmetry_coeff'])
        groove = float(request.form['groove'])

        # Calculate additional feature
        length_width_ratio = length / width if width != 0 else 0

        # Create DataFrame for model prediction
        features_df = pd.DataFrame({
            'Area': [area],
            'Perimeter': [perimeter],
            'Compactness': [compactness],
            'Length': [length],
            'Width': [width],
            'AsymmetryCoeff': [asymmetry_coeff],
            'Groove': [groove],
            'Length_Width_Ratio': [length_width_ratio]
        })

        print(f"Prediction DataFrame: {features_df}")

        # Perform prediction
        prediction = int(model.predict(features_df)[0])
        print(f"Prediction: {prediction}")

        return jsonify({"prediction": prediction})

    except Exception as e:
        print(f"Error during wheat prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)})

# Health check route
@wileen_app.route('/check')
def check():
    return jsonify({
        "status": "Wheat app is running",
        "model_loaded": model is not None
    })

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Wheat Flask app on port {port}")
    wileen_app.run(host='0.0.0.0', port=port, debug=True)
