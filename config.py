import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the application"""
    
    # Hugging Face API configuration
    HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')  # Load from environment variable
    
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Database configuration (if needed)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    
    # File upload configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
    
    # Model configuration
    MODEL_FOLDER = 'model'
    BEST_MODEL_FILE = 'best_predictor.pkl'
    SCALER_FILE = 'scaler.pkl'
    LABEL_ENCODER_FILE = 'label_encoder.pkl'
    
    @staticmethod
    def get_huggingface_headers():
        """Get headers for Hugging Face API calls"""
        if Config.HUGGINGFACE_TOKEN:
            return {"Authorization": f"Bearer {Config.HUGGINGFACE_TOKEN}"}
        return {}
    
    @staticmethod
    def is_configured():
        """Check if essential configuration is set up"""
        return bool(Config.HUGGINGFACE_TOKEN) 