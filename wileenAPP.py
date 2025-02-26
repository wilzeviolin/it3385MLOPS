import pickle
import numpy as np
import os
from flask import Flask, request, jsonify, render_template_string

# Initialize Flask app
app = Flask(__name__)

# Print environment information for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir(os.getcwd())}")

# Load the trained model
# Load the trained model
def load_model():
    try:
        # Check multiple possible locations for the model files
        model_filenames = ['seed_pipeline.pkl', 'seed_type_classification.pkl']
        possible_locations = []
        
        for filename in model_filenames:
            # Simple relative path (most likely on Render)
            possible_locations.append(filename)
            
            # Current working directory
            possible_locations.append(os.path.join(os.getcwd(), filename))
            
            # Script directory
            possible_locations.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename))
            
            # Parent directory
            possible_locations.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename))
        
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

# Feature ranges for guidance
FEATURE_RANGES = {
    'area': {'min': 10.59, 'max': 21.18},
    'perimeter': {'min': 12.41, 'max': 17.25},
    'compactness': {'min': 0.8081, 'max': 0.9183},
    'length': {'min': 4.899, 'max': 6.675},
    'width': {'min': 2.63, 'max': 4.033},
    'asymmetry_coeff': {'min': 0.7651, 'max': 8.315},
    'groove': {'min': 4.519, 'max': 6.55}
}

# HTML template with CSS included
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Wheat Type Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #336699;
            text-align: center;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .range-info {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9f7ef;
            border-radius: 4px;
            display: {{ 'block' if prediction is not none else 'none' }};
        }
        .error {
            margin-top: 20px;
            padding: 15px;
            background-color: #ffdddd;
            border-radius: 4px;
            display: {{ 'block' if error else 'none' }};
        }
        .wheat-types {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #336699;
        }
        .model-status {
            text-align: center;
            padding: 8px;
            margin-bottom: 20px;
            border-radius: 4px;
            font-weight: bold;
        }
        .status-ok {
            background-color: #d4edda;
            color: #155724;
        }
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
</head>
<body>
    <h1>Wheat Type Classifier</h1>
    
    <div class="model-status {{ 'status-ok' if model_loaded else 'status-error' }}">
        Model Status: {{ 'Loaded Successfully' if model_loaded else 'Error Loading Model' }}
    </div>
    
    <div class="container">
        <form action="/" method="post">
            <div class="form-group">
                <label for="area">Area:</label>
                <input type="number" id="area" name="area" step="0.01" required 
                       value="{{ request.form.get('area', '') }}"
                       min="{{ ranges['area']['min'] }}" max="{{ ranges['area']['max'] }}">
                <div class="range-info">Valid range: {{ ranges['area']['min'] }} to {{ ranges['area']['max'] }}</div>
            </div>
            
            <div class="form-group">
                <label for="perimeter">Perimeter:</label>
                <input type="number" id="perimeter" name="perimeter" step="0.01" required 
                       value="{{ request.form.get('perimeter', '') }}"
                       min="{{ ranges['perimeter']['min'] }}" max="{{ ranges['perimeter']['max'] }}">
                <div class="range-info">Valid range: {{ ranges['perimeter']['min'] }} to {{ ranges['perimeter']['max'] }}</div>
            </div>
            
            <div class="form-group">
                <label for="compactness">Compactness:</label>
                <input type="number" id="compactness" name="compactness" step="0.0001" required 
                       value="{{ request.form.get('compactness', '') }}"
                       min="{{ ranges['compactness']['min'] }}" max="{{ ranges['compactness']['max'] }}">
                <div class="range-info">Valid range: {{ ranges['compactness']['min'] }} to {{ ranges['compactness']['max'] }}</div>
            </div>
            
            <div class="form-group">
                <label for="length">Length:</label>
                <input type="number" id="length" name="length" step="0.01" required 
                       value="{{ request.form.get('length', '') }}"
                       min="{{ ranges['length']['min'] }}" max="{{ ranges['length']['max'] }}">
                <div class="range-info">Valid range: {{ ranges['length']['min'] }} to {{ ranges['length']['max'] }}</div>
            </div>
            
            <div class="form-group">
                <label for="width">Width:</label>
                <input type="number" id="width" name="width" step="0.01" required 
                       value="{{ request.form.get('width', '') }}"
                       min="{{ ranges['width']['min'] }}" max="{{ ranges['width']['max'] }}">
                <div class="range-info">Valid range: {{ ranges['width']['min'] }} to {{ ranges['width']['max'] }}</div>
            </div>
            
            <div class="form-group">
                <label for="asymmetry_coeff">Asymmetry Coefficient:</label>
                <input type="number" id="asymmetry_coeff" name="asymmetry_coeff" step="0.01" required 
                       value="{{ request.form.get('asymmetry_coeff', '') }}"
                       min="{{ ranges['asymmetry_coeff']['min'] }}" max="{{ ranges['asymmetry_coeff']['max'] }}">
                <div class="range-info">Valid range: {{ ranges['asymmetry_coeff']['min'] }} to {{ ranges['asymmetry_coeff']['max'] }}</div>
            </div>
            
            <div class="form-group">
                <label for="groove">Groove:</label>
                <input type="number" id="groove" name="groove" step="0.01" required 
                       value="{{ request.form.get('groove', '') }}"
                       min="{{ ranges['groove']['min'] }}" max="{{ ranges['groove']['max'] }}">
                <div class="range-info">Valid range: {{ ranges['groove']['min'] }} to {{ ranges['groove']['max'] }}</div>
            </div>
            
            <button type="submit" {{ 'disabled' if not model_loaded else '' }}>
                {{ 'Model Not Loaded - Please Fix' if not model_loaded else 'Predict Wheat Type' }}
            </button>
        </form>
        
        <div class="result">
            <h3>Prediction Result:</h3>
            <p><strong>Predicted Wheat Type: {{ prediction }}</strong></p>
        </div>
        
        <div class="error">
            <h3>Error:</h3>
            <p>{{ error }}</p>
            <p>Make sure the model file 'seed_type_classification.pkl' exists in the application directory.</p>
        </div>
        
        <div class="wheat-types">
            <h3>Wheat Type Reference:</h3>
            <p><strong>Type 1:</strong> Kama wheat</p>
            <p><strong>Type 2:</strong> Rosa wheat</p>
            <p><strong>Type 3:</strong> Canadian wheat</p>
        </div>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    model_loaded = model is not None

    if not model_loaded:
        error = "Model is not loaded. Check server logs for more details."

    if request.method == 'POST':
        if model_loaded:
            try:
                # Extract features from form data
                area = float(request.form['area'])
                perimeter = float(request.form['perimeter'])
                compactness = float(request.form['compactness'])
                length = float(request.form['length'])
                width = float(request.form['width'])
                asymmetry_coeff = float(request.form['asymmetry_coeff'])
                groove = float(request.form['groove'])
                
                # Prepare features for prediction
                features = np.array([[area, perimeter, compactness, length, width, asymmetry_coeff, groove]])
                
                # Make the prediction
                prediction = int(model.predict(features)[0])
            except Exception as e:
                error = f"Error making prediction: {str(e)}"
        else:
            error = "Model is not loaded. Cannot make predictions."

    return render_template_string(HTML_TEMPLATE, 
                                  prediction=prediction, 
                                  error=error, 
                                  model_loaded=model_loaded, 
                                  ranges=FEATURE_RANGES)

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
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not provided
    print(f"Model loaded: {model is not None}")
    if model is None:
        print("WARNING: Model failed to load. Check if 'seed_type_classification.pkl' exists.")
    app.run(host='0.0.0.0', port=port, debug=True)
