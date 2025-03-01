from flask import Flask
from wileenAPP import wileen_app
from roanne_carapp import roanne_app

main_app = Flask(__name__, template_folder='../templates')

# Register Wileen Routes
@main_app.route('/')
def home():
    return "Welcome to the Merged Flask App 🔥"

@main_app.route('/wheat')
def wheat():
    return wileen_app.view_functions['home_page']()

@main_app.route('/wheat/process', methods=['POST'])
def wheat_process():
    return wileen_app.view_functions['process_form']()

# Register Roanne Routes
@main_app.route('/car')
def car():
    return roanne_app.view_functions['car_page']()

@main_app.route('/car/predict', methods=['POST'])
def car_predict():
    return roanne_app.view_functions['predict']()

if __name__ == '__main__':
    main_app.run(host='0.0.0.0', port=5000, debug=True)
