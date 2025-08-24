
import logging
from flask import Flask
from app.routes.main_routes import main_bp
from app.routes.api_routes import api_bp


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(api_bp)


if __name__ == '__main__':
    logger.info("Starting Student Result Predictor...")
    app.run(debug=True, host='0.0.0.0', port=5000)