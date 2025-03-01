from flask import Flask, render_template
from wileenAPP import wileen_app
from roanne_carapp import roanne_app
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


main_app = Flask(__name__, template_folder='../templates')

@main_app.route('/')
def home():
    return render_template('home.html')

@main_app.route('/wheat')
def wheat():
    return wileen_app.view_functions['home_page']()

@main_app.route('/wheat/process', methods=['POST'])
def wheat_process():
    return wileen_app.view_functions['process_form']()

@main_app.route('/car')
def car():
    return roanne_app.view_functions['home']()

@main_app.route('/car/predict', methods=['POST'])
def car_predict():
    return roanne_app.view_functions['predict']()

if __name__ == '__main__':
    main_app.run(host='0.0.0.0', port=5000, debug=True)
