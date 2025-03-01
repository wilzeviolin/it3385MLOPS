from flask import Flask
from wileenAPP import wileen_app
from roanne_carapp import roanne_app

main_app = Flask(__name__, template_folder='../templates')

# Register Blueprints
main_app.register_blueprint(wileen_app, url_prefix='/wheat')
main_app.register_blueprint(roanne_app, url_prefix='/car')

@main_app.route('/')
def home():
    return "Welcome to the Merged Flask App 🔥"

if __name__ == '__main__':
    main_app.run(host='0.0.0.0', port=5000, debug=True)
