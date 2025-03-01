import pickle
import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import traceback

wileen_app = Flask(__name__, template_folder='../templates')

# Force Compatibility Model Loader
def custom_load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file, fix_imports=True, encoding='latin1')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the trained model
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, 'artifacts', 'seed_pipeline.pkl')

        print(f"Loading model from: {model_path}")
        if os.path.exists(model_path):
            loaded_model = custom_load_model(model_path)
            print("Model loaded successfully")
            return loaded_model
        else:
            print("Model file not found")
            return None

    except Exception as e:
        print(f"Model loading error: {e}")
        print(traceback.format_exc())
        return None

model = load_model()

@wileen_app.route('/')
def home_page():
    return render_template('wheat.html')

@wileen_app.route('/process', methods=['POST'])
def process_form():
    print("Processing wheat form...")

    if model is None:
        return jsonify({"error": "Model not loaded"})

    try:
        area = float(request.form['area'])
        perimeter = float(request.form['perimeter'])
        compactness = float(request.form['compactness'])
        length = float(request.form['length'])
        width = float(request.form['width'])
        asymmetry_coeff = float(request.form['asymmetry_coeff'])
        groove = float(request.form['groove'])

        length_width_ratio = length / width if width != 0 else 0

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

        prediction = int(model.predict(features_df)[0])
        print(f"Prediction: {prediction}")
        return jsonify({"prediction": prediction})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)})

@wileen_app.route('/check')
def check():
    return jsonify({
        "status": "Wheat app is running",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    wileen_app.run(host='0.0.0.0', port=10000, debug=True)
