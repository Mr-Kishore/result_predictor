import pickle
import os

def check_training_results():
    """Check and display the training results"""
    print("Student Result Predictor - Training Results")
    print("="*50)
    
    # Check if model files exist
    model_files = [
        'model/best_predictor.pkl',
        'model/all_models.pkl', 
        'model/evaluation_results.pkl',
        'model/model_info.pkl',
        'model/scaler.pkl',
        'model/label_encoder.pkl'
    ]
    
    print("Checking model files:")
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size} bytes)")
        else:
            print(f"✗ {file_path} - MISSING")
    
    print("\n" + "="*50)
    
    # Load and display model info
    try:
        with open('model/model_info.pkl', 'rb') as f:
            info = pickle.load(f)
        
        print("TRAINING RESULTS:")
        print(f"Best Model: {info['best_model']}")
        print(f"Best Cross-Validation Score: {info['best_score']:.4f}")
        
        print("\nModel Performance Comparison:")
        print("-" * 40)
        for model_name, metrics in info['model_comparison'].items():
            print(f"{model_name.upper():<20} | Accuracy: {metrics['accuracy']:.4f} | CV Score: {metrics['cv_score']:.4f}")
        
        print(f"\nFeature Columns ({len(info['feature_columns'])}):")
        for i, feature in enumerate(info['feature_columns'], 1):
            print(f"{i:2d}. {feature}")
            
    except Exception as e:
        print(f"Error loading model info: {e}")
    
    print("\n" + "="*50)
    print("To run the Flask application: python app.py")
    print("To retrain models: python train_model.py")

if __name__ == "__main__":
    check_training_results() 