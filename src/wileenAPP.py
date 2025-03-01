import pickle
import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template

wileen_app = Flask(__name__, template_folder='../templates')  # Correctly initialize the app

# Your route definitions here
@wileen_app.route('/')
def home():
    return "Wileen App Home"

def load_model():
    try:
        # Use absolute path with __file__ to find the model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, 'artifacts', 'seed_pipeline.pkl')
        
        print(f"Attempting to load model from: {model_path}")
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
                print("Model loaded successfully")
                return model
        else:
            print(f"Model file not found at {model_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model at startup
model = load_model()

@wileen_app.route('/', methods=['GET'])
def home_page():
    return render_template('wheat.html')

@wileen_app.route('/process', methods=['POST'])
def process_form():
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
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

@wileen_app.before_request
def check_model():
    global model
    if model is None:
        model = load_model()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Flask app on port {port}")
    wileen_app.run(host='0.0.0.0', port=port, debug=False)

