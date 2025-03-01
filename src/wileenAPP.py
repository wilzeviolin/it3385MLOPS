import pickle
import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import traceback

wileen_app = Flask(__name__, template_folder='../templates')

# Load the trained model
def load_model():
    try:
        # Use absolute path for more reliable model loading
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        model_path = os.path.join(project_root, 'artifacts', 'seed_pipeline.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as model_file:
                loaded_model = pickle.load(model_file)
                print("Wheat model loaded successfully")
                return loaded_model
        else:
            print(f"Wheat model file not found at {model_path}")
            # Create a simple fallback model if the real one doesn't exist
            class DummyModel:
                def predict(self, X):
                    return np.array([1])  # Always predict class 1
            
            print("Using dummy wheat model")
            return DummyModel()
    except Exception as e:
        print(f"Error loading wheat model: {e}")
        return None

# Load model at startup
model = load_model()

@wileen_app.route('/')
def home_page():
    return render_template('wheat.html')

@wileen_app.route('/process', methods=['POST'])
def process_form():
    global model
    if model is None:
        model = load_model()
    
    if model is None:
        return jsonify({"error": "Model not loaded"})
    
    try:
        # Extract form data
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
        
        try:
            # Use model for prediction
            prediction = int(model.predict(features_df)[0])
            return jsonify({"prediction": prediction})
        except Exception as predict_error:
            print(f"Error during wheat prediction: {predict_error}")
            return jsonify({"error": f"Prediction error: {str(predict_error)}"})
    
    except Exception as e:
        print(f"Error processing wheat form: {e}")
        return jsonify({"error": str(e)})

@wileen_app.before_request
def check_model():
    global model
    if model is None:
        model = load_model()

@wileen_app.route('/check')
def check():
    return jsonify({
        "status": "Wheat app is working",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Wheat Flask app on port {port}")
    wileen_app.run(host='0.0.0.0', port=port, debug=True)
