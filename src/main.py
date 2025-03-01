from flask import Flask, render_template, request, jsonify
import sys
import os
import traceback

# Add parent directory to path to ensure imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the apps after adding the path
from wileenAPP import wileen_app, load_model as load_wheat_model, model as wheat_model
from roanne_carapp import roanne_app

main_app = Flask(__name__, template_folder='../templates')

@main_app.route('/')
def home():
    return render_template('home.html')

@main_app.route('/wheat')
def wheat():
    try:
        return wileen_app.view_functions['home_page']()
    except Exception as e:
        print(f"Error routing to wheat page: {e}")
        print(traceback.format_exc())
        return f"Error loading wheat page: {str(e)}"

@main_app.route('/wheat/process', methods=['POST'])
def wheat_process():
    try:
        # Extract form data directly
        form_data = request.form.to_dict()
        print(f"Wheat process form data: {form_data}")
        
        # Process the wheat data directly
        if wheat_model is None:
            wheat_model_loaded = load_wheat_model()
            print(f"Loaded wheat model: {wheat_model_loaded}")
        
        area = float(form_data['area'])
        perimeter = float(form_data['perimeter'])
        compactness = float(form_data['compactness'])
        length = float(form_data['length'])
        width = float(form_data['width'])
        asymmetry_coeff = float(form_data['asymmetry_coeff'])
        groove = float(form_data['groove'])
        length_width_ratio = length / width if width != 0 else 0
        
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
        
        # Use the wheat model to make a prediction
        if wheat_model is not None:
            prediction = int(wheat_model.predict(features_df)[0])
            return jsonify({"prediction": prediction})
        else:
            # Fallback to a simple prediction if model is not available
            return jsonify({"prediction": 1, "note": "Using fallback prediction (model not available)"})
    
    except Exception as e:
        print(f"Error in wheat_process: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error processing wheat data: {str(e)}"})

@main_app.route('/car')
def car():
    try:
        return roanne_app.view_functions['home']()
    except Exception as e:
        print(f"Error routing to car page: {e}")
        print(traceback.format_exc())
        return f"Error loading car page: {str(e)}"

@main_app.route('/car/predict', methods=['POST'])
def car_predict():
    try:
        # Process car prediction directly
        form_data = request.form.to_dict()
        print(f"Car predict form data: {form_data}")
        
        # Calculate prediction using the formula from roanne_carapp.py
        user_input = {
            "Year": int(form_data['year']),
            "Kilometers_Driven": float(form_data['kilometers_driven']),
        }
        
        # Simple prediction formula
        base_value = 15.0  # Base value in lakhs
        year_factor = (user_input["Year"] - 2010) * 0.5  # 0.5 lakhs per year after 2010
        mileage_discount = user_input["Kilometers_Driven"] / 10000 * 0.2  # 0.2 lakhs per 10k km
        
        prediction = base_value + year_factor - mileage_discount
        prediction = max(prediction, 1.0)  # Minimum price of 1 lakh
        
        return jsonify({"Predicted Price (INR Lakhs)": round(prediction, 2)})
        
    except Exception as e:
        print(f"Error in car_predict: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error predicting car price: {str(e)}"})

# Add this new route to handle direct /predict requests
@main_app.route('/predict', methods=['POST'])
def predict_direct():
    try:
        form_data = request.form.to_dict()
        print(f"Direct predict route called with data: {form_data}")
        
        # Determine which prediction service to use based on the form fields
        if any(key in form_data for key in ['year', 'kilometers_driven', 'brand_model']):
            print("Routing to car prediction")
            return car_predict()
        elif any(key in form_data for key in ['area', 'perimeter', 'compactness', 'length', 'width']):
            print("Routing to wheat prediction")
            return wheat_process()
        else:
            print("Could not determine prediction type")
            return jsonify({
                "error": "Could not determine prediction type. Please include relevant fields for car or wheat prediction."
            })
    except Exception as e:
        print(f"Error in direct predict route: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction error: {str(e)}"})

# Add this route to handle direct /process requests for wheat
@main_app.route('/process', methods=['POST'])
def process_direct():
    try:
        print("Direct process route called - redirecting to wheat_process")
        return wheat_process()
    except Exception as e:
        print(f"Error in direct process route: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Wheat processing error: {str(e)}"})

# Add a diagnostic route
@main_app.route('/debug')
def debug():
    try:
        # Verify paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        artifact_dir = os.path.join(project_root, "artifacts")
        
        # Check if artifact directory exists
        artifact_dir_exists = os.path.exists(artifact_dir)
        
        # Get list of files in artifacts directory if it exists
        artifact_files = os.listdir(artifact_dir) if artifact_dir_exists else []
        
        # Check model files
        wheat_model_path = os.path.join(artifact_dir, "seed_pipeline.pkl")
        car_model_path_pkl = os.path.join(artifact_dir, "used_car_price_model.pkl")
        car_model_path_joblib = os.path.join(artifact_dir, "used_car_price_model.joblib")
        
        debug_info = {
            "current_directory": os.getcwd(),
            "project_root": project_root,
            "artifact_directory": artifact_dir,
            "artifact_dir_exists": artifact_dir_exists,
            "artifact_files": artifact_files,
            "wheat_model_exists": os.path.exists(wheat_model_path),
            "car_model_pkl_exists": os.path.exists(car_model_path_pkl),
            "car_model_joblib_exists": os.path.exists(car_model_path_joblib),
            "routes": [rule.rule for rule in main_app.url_map.iter_rules()]
        }
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": str(e)})

@main_app.route('/test-form', methods=['GET', 'POST'])
def test_form():
    if request.method == 'GET':
        return '''
        <form method="POST">
            <input type="text" name="test_value">
            <input type="submit" value="Test">
        </form>
        '''
    else:
        try:
            test_value = request.form.get('test_value', 'No value')
            return f"Received form value: {test_value}"
        except Exception as e:
            return f"Error: {str(e)}"

# Add an error handler for 404 errors that returns JSON instead of HTML
@main_app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found. Available endpoints include /car/predict, /wheat/process, /predict, and /process"}), 404

if __name__ == '__main__':
    # Print debug info at startup
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Check if templates directory exists
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
    print(f"Template directory: {template_dir}")
    print(f"Template directory exists: {os.path.exists(template_dir)}")
    
    # Print available routes
    print("Available routes:")
    for rule in main_app.url_map.iter_rules():
        print(f"- {rule.rule} ({', '.join(rule.methods)})")
    
    main_app.run(host='0.0.0.0', port=5000, debug=True)
