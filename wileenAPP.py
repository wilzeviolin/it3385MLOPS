import pickle
import numpy as np
import os
from flask import Flask, request, jsonify, render_template_string

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
def load_model():
    try:
        with open('seed_type_classification.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            return model
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Load the model once when the app starts
model = load_model()

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
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
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
        .wheat-types {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #336699;
        }
    </style>
</head>
<body>
    <h1>Wheat Type Classifier</h1>
    
    <div class="container">
        <form action="/" method="post">
            <div class="form-group">
                <label for="area">Area:</label>
                <input type="number" id="area" name="area" step="0.01" required value="{{ request.form.get('area', '') }}">
            </div>
            
            <div class="form-group">
                <label for="perimeter">Perimeter:</label>
                <input type="number" id="perimeter" name="perimeter" step="0.01" required value="{{ request.form.get('perimeter', '') }}">
            </div>
            
            <div class="form-group">
                <label for="compactness">Compactness:</label>
                <input type="number" id="compactness" name="compactness" step="0.0001" required value="{{ request.form.get('compactness', '') }}">
            </div>
            
            <div class="form-group">
                <label for="length">Length:</label>
                <input type="number" id="length" name="length" step="0.01" required value="{{ request.form.get('length', '') }}">
            </div>
            
            <div class="form-group">
                <label for="width">Width:</label>
                <input type="number" id="width" name="width" step="0.01" required value="{{ request.form.get('width', '') }}">
            </div>
            
            <div class="form-group">
                <label for="asymmetry_coeff">Asymmetry Coefficient:</label>
                <input type="number" id="asymmetry_coeff" name="asymmetry_coeff" step="0.01" required value="{{ request.form.get('asymmetry_coeff', '') }}">
            </div>
            
            <div class="form-group">
                <label for="groove">Groove:</label>
                <input type="number" id="groove" name="groove" step="0.01" required value="{{ request.form.get('groove', '') }}">
            </div>
            
            <button type="submit">Predict Wheat Type</button>
        </form>
        
        <div class="result">
            <h3>Prediction Result:</h3>
            <p><strong>Predicted Wheat Type: {{ prediction }}</strong></p>
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
    
    if request.method == 'POST':
        try:
            # Extract the features
            area = float(request.form['area'])
            perimeter = float(request.form['perimeter'])
            compactness = float(request.form['compactness'])
            length = float(request.form['length'])
            width = float(request.form['width'])
            asymmetry_coeff = float(request.form['asymmetry_coeff'])
            groove = float(request.form['groove'])
            
            # Prepare the input features as a numpy array for prediction
            features = np.array([[area, perimeter, compactness, length, width, asymmetry_coeff, groove]])
            
            # Make the prediction
            prediction = int(model.predict(features)[0])
            
            # Map to wheat type names for better display
            wheat_types = {0: "Type 1 (Kama)", 1: "Type 2 (Rosa)", 2: "Type 3 (Canadian)"}
            prediction = wheat_types.get(prediction, f"Unknown (Type {prediction})")
            
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

# API endpoint still available for programmatic access
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not provided
    app.run(host='0.0.0.0', port=port, debug=True)
