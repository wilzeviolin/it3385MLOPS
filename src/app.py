from flask import Flask, render_template, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model

app = Flask(__name__)

# Load the trained model
model = load_model("best_pycaret_model")  # Ensure this file exists in deployment

# Expected columns based on PyCaret training
expected_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Method', 'Seller', 'Date', 
                    'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
                    'BuildingArea', 'YearBuilt', 'CouncilArea', 'Latitude', 'Longitude', 
                    'Region', 'Propertycount']

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Ensure all required columns exist
        for col in expected_columns:
            if col not in df.columns:
                df[col] = "Unknown" if df[col].dtype == object else 0  # Default value

        # Convert numeric fields
        numeric_columns = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 
                           'Landsize', 'BuildingArea', 'YearBuilt', 'Latitude', 'Longitude', 
                           'Propertycount']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  # Convert to numbers

        # Make prediction
        prediction = predict_model(model, data=df)
        predicted_price = prediction["Label"].tolist()

        return jsonify({"prediction": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
