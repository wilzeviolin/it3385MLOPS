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

    <div class="container">
        <form action="/" method="post">
            <div class="form-group">
                <label for="area">Area:</label>
                <input type="number" id="area" name="area" step="1" required 
                       value="{{ request.form.get('area', '') }}">
                <div class="range-info">Suggested range: 10 to 200</div> <!-- Rough range based on previous data -->
            </div>

            <div class="form-group">
                <label for="perimeter">Perimeter:</label>
                <input type="number" id="perimeter" name="perimeter" step="1" required 
                       value="{{ request.form.get('perimeter', '') }}">
                <div class="range-info">Suggested range: 30 to 100</div> <!-- Rough range based on previous data -->
            </div>

            <div class="form-group">
                <label for="compactness">Compactness:</label>
                <input type="number" id="compactness" name="compactness" step="1" required 
                       value="{{ request.form.get('compactness', '') }}">
                <div class="range-info">Suggested range: 0 to 1</div> <!-- Rough range based on previous data -->
            </div>

            <div class="form-group">
                <label for="length">Length:</label>
                <input type="number" id="length" name="length" step="1" required 
                       value="{{ request.form.get('length', '') }}">
                <div class="range-info">Suggested range: 4 to 8</div> <!-- Rough range based on previous data -->
            </div>

            <div class="form-group">
                <label for="width">Width:</label>
                <input type="number" id="width" name="width" step="1" required 
                       value="{{ request.form.get('width', '') }}">
                <div class="range-info">Suggested range: 2 to 5</div> <!-- Rough range based on previous data -->
            </div>

            <div class="form-group">
                <label for="asymmetry_coeff">Asymmetry Coefficient:</label>
                <input type="number" id="asymmetry_coeff" name="asymmetry_coeff" step="1" required 
                       value="{{ request.form.get('asymmetry_coeff', '') }}">
                <div class="range-info">Suggested range: 2 to 6</div> <!-- Rough range based on previous data -->
            </div>

            <div class="form-group">
                <label for="groove">Groove:</label>
                <input type="number" id="groove" name="groove" step="1" required 
                       value="{{ request.form.get('groove', '') }}">
                <div class="range-info">Suggested range: 3 to 10</div> <!-- Rough range based on previous data -->
            </div>

            <button type="submit" {{ 'disabled' if not model_loaded else '' }}>
                {{ 'Model Not Loaded - Please Fix' if not model_loaded else 'Predict Wheat Type' }}
            </button>
        </form>

        {% if prediction %}
        <div class="result">
            <h3>Prediction Result:</h3>
            <p><strong>Predicted Wheat Type: {{ prediction }}</strong></p>
        </div>
        {% endif %}

        {% if error %}
        <div class="error">
            <h3>Error:</h3>
            <p>{{ error }}</p>
            <p>Ensure the model file 'seed_type_classification.pkl' exists in the application directory.</p>
        </div>
        {% endif %}

        <div class="wheat-types">
            <h3>Wheat Type Reference:</h3>
            <p><strong>Type 1: Kama wheat</strong></p>
            <p>Kama wheat is typically grown in temperate climates and is known for its high protein content and excellent baking quality. It is commonly used in making bread and other baked goods.</p>
            
            <p><strong>Type 2: Rosa wheat</strong></p>
            <p>Rosa wheat is characterized by its lighter color and high yield. It is often used for making pastries and cakes due to its delicate texture.</p>

            <p><strong>Type 3: Canadian wheat</strong></p>
            <p>Canadian wheat is widely recognized for its strong gluten content, making it ideal for bread and pasta production. It is highly valued for its versatility and high milling yield.</p>
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
                
                # Calculate the length-width ratio that the model expects
                length_width_ratio = length / width if width != 0 else 0
                
                # Create a pandas DataFrame with the expected column names
                import pandas as pd
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
        import pandas as pd
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
        
        # Return the prediction as a response
        return jsonify({"predicted_wheat_type": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Ensure port matches Render's config

