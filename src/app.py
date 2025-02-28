from flask import Flask, render_template, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model
import os

app = Flask(__name__)

# Path to artifacts folder
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '../artifacts', 'best_pycaret_model')

# Load the trained model
model = load_model(ARTIFACTS_PATH)  # ✅ Access from artifacts folder

# Expected columns based on PyCaret training
expected_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Method', 'Seller', 'Date', 
                    'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
                    'BuildingArea', 'YearBuilt', 'CouncilArea', 'Latitude', 'Longitude', 
                    'Region', 'Propertycount']

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Add missing columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = "Unknown" if df[col].dtype == object else 0

        # Convert numeric columns
        numeric_columns = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
                           'Landsize', 'BuildingArea', 'YearBuilt', 'Latitude', 'Longitude', 
                           'Propertycount']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Make prediction
        prediction = predict_model(model, data=df)
        predicted_price = prediction["Label"].tolist()

        return jsonify({"prediction": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
