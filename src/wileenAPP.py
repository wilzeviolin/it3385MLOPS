import pickle
import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, render_template

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')

# Print environment information for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir(os.getcwd())}")

# Load the trained model
def load_model():
    try:
        # Check for model files in the artifacts directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        artifacts_dir = os.path.join(project_root, 'artifacts')
        model_filenames = ['seed_pipeline.pkl', 'seed_type_classification.pkl']
        possible_locations = []
        
        # Build list of possible file locations
        for filename in model_filenames:
            # Artifacts directory
            possible_locations.append(os.path.join(artifacts_dir, filename))
        
        for location in possible_locations:
            print(f"Checking for model at: {location}")
            if os.path.exists(location):
                print(f"Found model at: {location}")
                
                # Try different loading methods
                try:
                    # Method 1: Standard loading
                    with open(location, 'rb') as model_file:
                        model = pickle.load(model_file)
                        print(f"Model loaded successfully from {location}!")
                        return model
                except Exception as e1:
                    print(f"Standard loading failed from {location}: {e1}")
                    try:
                        # Method 2: Try with latin1 encoding
                        with open(location, 'rb') as model_file:
                            model = pickle.load(model_file, encoding='latin1')
                            print(f"Model loaded with latin1 encoding from {location}!")
                            return model
                    except Exception as e2:
                        print(f"Latin1 encoding failed from {location}: {e2}")
                        try:
                            # Method 3: Try with bytes encoding
                            with open(location, 'rb') as model_file:
                                model = pickle.load(model_file, encoding='bytes')
                                print(f"Model loaded with bytes encoding from {location}!")
                                return model
                        except Exception as e3:
                            print(f"Bytes encoding failed from {location}: {e3}")
        
        print("Could not find or load model from any of the expected locations")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

# Load the model once when the app starts
model = load_model()

@app.route('/', methods=['GET'])
def home_page():
    # Use Flask's render_template to serve the HTML file from templates directory
    return render_template('wheat.html')

@app.route('/process', methods=['POST'])
def process_form():
    prediction = None
    error = None
    model_loaded = model is not None

    if not model_loaded:
        return jsonify({"error": "Model is not loaded. Check server logs for more details."})

    try:
        # Extract features from form data
        area = float(request.form['area'])
        perimeter = float(request.form['perimeter'])
        compactness = float(request.form['compactness'])
        length = float(request.form['length'])
        width = float(request.form['width'])
        asymmetry_coeff = float(request.form['asymmetry_coeff'])
        groove = float(request.form['groove'])
        
        # Calculate the length-width ratio that the model expects
        length_width_ratio = length / width if width != 0 else 0
        
        # Create a pandas DataFrame with the expected column names
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
        
        # Make the prediction using the DataFrame
        prediction = int(model.predict(features_df)[0])
            
        # Return only the prediction without confidence
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": f"Error making prediction: {str(e)}"})

# Try to initialize model again if it failed to load at startup
@app.before_request
def check_model():
    global model
    if model is None:
        model = load_model()

# API endpoint still available for programmatic access
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        # Get input features from the request
        data = request.get_json(force=True)
        
        # Extract the features
        area = data.get('Area', data.get('area'))
        perimeter = data.get('Perimeter', data.get('perimeter'))
        compactness = data.get('Compactness', data.get('compactness'))
        length = data.get('Length', data.get('length'))
        width = data.get('Width', data.get('width'))
        asymmetry_coeff = data.get('AsymmetryCoeff', data.get('asymmetry_coeff'))
        groove = data.get('Groove', data.get('groove'))
        
        # Calculate length-width ratio
        length_width_ratio = length / width if width != 0 else 0
        
        # Create DataFrame with proper column names
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
        
        # Make the prediction
        prediction = int(model.predict(features_df)[0])
        
        # Return only the prediction without confidence
        return jsonify({"predicted_wheat_type": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Try a different port if 5000 is blocked
    print(f"Model loaded: {model is not None}")
    if model is None:
        print("WARNING: Model failed to load. Check if model files exist.")
    else:
        print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
