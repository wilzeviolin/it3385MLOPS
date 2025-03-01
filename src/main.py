from flask import Flask, render_template, request, jsonify
import os

# Import the apps
from wileenAPP import wileen_app
from roanne_carapp import roanne_app

main_app = Flask(__name__, template_folder='../templates')

@main_app.route('/')
def home():
    return render_template('home.html')

@main_app.route('/wheat')
def wheat():
    return wileen_app.view_functions['home_page']()

@main_app.route('/wheat/process', methods=['POST'])
def wheat_process():
    # Call the original process_form function from wileen_app
    return wileen_app.view_functions['process_form']()

@main_app.route('/car')
def car():
    return roanne_app.view_functions['home']()

@main_app.route('/car/predict', methods=['POST'])
def car_predict():
    # Call the original predict function from roanne_app
    return roanne_app.view_functions['predict']()

# General predict endpoint that routes to the appropriate prediction function
@main_app.route('/predict', methods=['POST'])
def predict_direct():
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

# General process endpoint for wheat
@main_app.route('/process', methods=['POST'])
def process_direct():
    return wheat_process()

# Error handler for 404 errors
@main_app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found. Available endpoints include /car/predict, /wheat/process, /predict, and /process"}), 404

if __name__ == '__main__':
    main_app.run(host='0.0.0.0', port=5000, debug=True)
