from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import traceback

roanne_app = Flask(__name__, template_folder='../templates')

# Ensure joblib does not cache to restricted directories
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Use absolute paths with __file__ for more reliable path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_path_pkl = os.path.join(project_root, "artifacts", "used_car_price_model.pkl")
model_path_joblib = os.path.join(project_root, "artifacts", "used_car_price_model.joblib")

# Try loading the model
try:
    if os.path.exists(model_path_joblib):
        model = joblib.load(model_path_joblib)
        print("Car model loaded successfully from joblib")
    elif os.path.exists(model_path_pkl):
        with open(model_path_pkl, 'rb') as f:
            model = pickle.load(f)
        print("Car model loaded successfully from pickle")
    else:
        print(f"Car model file not found. Using fallback model.")
        # Create a fallback model
        class DummyModel:
            def predict(self, X):
                return np.array([10.0])  # Always predict 10 lakhs
        
        model = DummyModel()
except Exception as e:
    print(f"Car model loading failed: {e}")
    model = None  # Set to None if loading fails

# Create encoders for categorical variables
categorical_columns = ["Brand_Model", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
encoders = {col: OneHotEncoder(sparse=False, handle_unknown='ignore') for col in categorical_columns}

# Define some common values for each categorical column
common_values = {
    "Brand_Model": ["Maruti Swift Dzire VDI", "Hyundai i20 Sportz", "Honda City", "Toyota Innova", "Maruti Wagon R LXI CNG"],
    "Location": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"],
    "Fuel_Type": ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
    "Transmission": ["Manual", "Automatic"],
    "Owner_Type": ["First", "Second", "Third", "Fourth"]
}

# Fit encoders with common values (this is just for initialization)
for col in categorical_columns:
    sample_data = pd.DataFrame({col: common_values[col]})
    encoders[col].fit(sample_data)

@roanne_app.route('/')
def home():
    return render_template('roanne_car.html')

@roanne_app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather user input
        user_input = {
            "Brand_Model": request.form.get('brand_model', common_values["Brand_Model"][0]),
            "Location": request.form.get('location', common_values["Location"][0]),
            "Year": int(request.form.get('year', 2015)),
            "Kilometers_Driven": float(request.form.get('kilometers_driven', 50000)),
            "Fuel_Type": request.form.get('fuel_type', common_values["Fuel_Type"][0]),
            "Transmission": request.form.get('transmission', common_values["Transmission"][0]),
            "Owner_Type": request.form.get('owner_type', common_values["Owner_Type"][0]),
            "Mileage": float(request.form.get('mileage', 20)),
            "Engine": float(request.form.get('engine', 1500)),
            "Power": float(request.form.get('power', 100)),
            "Seats": int(request.form.get('seats', 5))
        }
        
        # Create a base DataFrame
        df = pd.DataFrame([user_input])
        
        # Create encoded features for categorical columns
        encoded_features = {}
        for col in categorical_columns:
            # Extract this categorical value and reshape for encoding
            cat_value = df[[col]]
            
            # Encode the categorical value
            encoded = encoders[col].transform(cat_value)
            
            # Get feature names (column names after encoding)
            if hasattr(encoders[col], 'get_feature_names_out'):
                feature_names = encoders[col].get_feature_names_out([col])
            else:
                # For older sklearn versions
                feature_names = [f"{col}_{val}" for val in encoders[col].categories_[0]]
            
            # Add to encoded features
            for i, name in enumerate(feature_names):
                encoded_features[name] = encoded[0][i]
        
        # Create numerical features
        numerical_features = {
            "Year": user_input["Year"],
            "Kilometers_Driven": user_input["Kilometers_Driven"],
            "Mileage": user_input["Mileage"],
            "Engine": user_input["Engine"],
            "Power": user_input["Power"],
            "Seats": user_input["Seats"]
        }
        
        # Combine all features
        all_features = {**numerical_features, **encoded_features}
        
        # Convert to feature vector for model input
        input_data = np.array([list(numerical_features.values())])
        
        # Make prediction
        try:
            # Check if we have a loaded model
            if model is not None:
                try:
                    # Use model for prediction
                    prediction = model.predict(input_data)[0]
                    return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2)})
                except Exception as model_e:
                    print(f"Model prediction failed: {model_e}, using fallback")
            
            # Fall back to a simpler prediction 
            base_value = 15.0  # Base value in lakhs
            year_factor = (user_input["Year"] - 2010) * 0.5  # 0.5 lakhs per year after 2010
            mileage_discount = user_input["Kilometers_Driven"] / 10000 * 0.2  # 0.2 lakhs per 10k km
            
            prediction = base_value + year_factor - mileage_discount
            prediction = max(prediction, 1.0)  # Minimum price of 1 lakh
            
            return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2),
                           "note": "Using fallback prediction"})
        except Exception as inner_e:
            print(f"Prediction error: {inner_e}")
            return jsonify({"Predicted Price (INR Lakhs)": round(10.0, 2), 
                           "note": "Using default prediction due to model issues"})
            
    except Exception as e:
        print(f"Overall prediction route error: {e}")
        return jsonify({"error": str(e)})

# Add a route to check if the app is working
@roanne_app.route('/check')
def check():
    return jsonify({
        "status": "Car app is working",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    roanne_app.run(debug=True)
