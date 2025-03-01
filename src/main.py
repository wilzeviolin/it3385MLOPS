from flask import Flask, render_template, request, jsonify
import sys
import os

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
        return f"Error loading wheat page: {str(e)}"

@main_app.route('/wheat/process', methods=['POST'])
def wheat_process():
    try:
        # Pass the request to the wheat process_form function
        return wileen_app.view_functions['process_form']()
    except Exception as e:
        print(f"Error in wheat_process: {e}")
        return jsonify({"error": f"Error processing wheat data: {str(e)}"})

@main_app.route('/car')
def car():
    try:
        return roanne_app.view_functions['home']()
    except Exception as e:
        print(f"Error routing to car page: {e}")
        return f"Error loading car page: {str(e)}"

@main_app.route('/car/predict', methods=['POST'])
def car_predict():
    try:
        # Pass the request to the car predict function
        return roanne_app.view_functions['predict']()
    except Exception as e:
        print(f"Error in car_predict: {e}")
        return jsonify({"error": f"Error predicting car price: {str(e)}"})

# Route to handle direct /predict requests
@main_app.route('/predict', methods=['POST'])
def predict_direct():
    try:
        form_data = request.form.to_dict()
        
        # Determine which prediction service to use based on the form fields
        if any(key in form_data for key in ['year', 'kilometers_driven', 'brand_model']):
            return car_predict()
        elif any(key in form_data for key in ['area', 'perimeter', 'compactness', 'length', 'width']):
            return wheat_process()
        else:
            return jsonify({
                "error": "Could not determine prediction type. Please include relevant fields for car or wheat prediction."
            })
    except Exception as e:
        print(f"Error in direct predict route: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"})

# Route to handle direct /process requests for wheat
@main_app.route('/process', methods=['POST'])
def process_direct():
    try:
        return wheat_process()
    except Exception as e:
        print(f"Error in direct process route: {e}")
        return jsonify({"error": f"Wheat processing error: {str(e)}"})

# Simplified debug route
@main_app.route('/debug')
def debug():
    try:
        # Verify paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        artifact_dir = os.path.join(project_root, "artifacts")
        
        # Get basic debug info
        debug_info = {
            "current_directory": os.getcwd(),
            "project_root": project_root,
            "wheat_model_loaded": wheat_model is not None,
            "available_routes": [rule.rule for rule in main_app.url_map.iter_rules()]
        }
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": str(e)})

# Add an error handler for 404 errors that returns JSON
@main_app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found. Available endpoints include /car/predict, /wheat/process, /predict, and /process"}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render uses PORT environment variable
    print(f"Starting server on port {port}...")
    main_app.run(host='0.0.0.0', port=port, debug=False)


