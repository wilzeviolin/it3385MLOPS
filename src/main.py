from flask import Flask
from wileenAPP import app as wileen_app
from app import app as second_app

main_app = Flask(__name__)

# Register Blueprints
main_app.register_blueprint(wileen_app, url_prefix='/wheat')
main_app.register_blueprint(second_app, url_prefix='/another')

@main_app.route('/')
def home():
    return "Welcome to the Merged Flask App 🔥"

if __name__ == '__main__':
    main_app.run(host='0.0.0.0', port=5000, debug=True)
