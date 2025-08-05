import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class StudentResultPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_score = 0
        
    def generate_sample_data(self, n_samples=1000):
        """Generate sample student data for training"""
        np.random.seed(42)
        
        # Generate realistic student features
        data = {
            'student_id': range(1, n_samples + 1),
            'attendance_percentage': np.random.normal(85, 15, n_samples).clip(0, 100),
            'assignment_marks': np.random.normal(75, 20, n_samples).clip(0, 100),
            'midterm_marks': np.random.normal(70, 25, n_samples).clip(0, 100),
            'final_exam_marks': np.random.normal(65, 30, n_samples).clip(0, 100),
            'study_hours_per_day': np.random.normal(3, 1.5, n_samples).clip(0, 8),
            'previous_semester_gpa': np.random.normal(3.0, 0.8, n_samples).clip(0, 4),
            'extracurricular_activities': np.random.choice([0, 1, 2, 3], n_samples),
            'family_income': np.random.normal(50000, 20000, n_samples).clip(20000, 100000),
            'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'age': np.random.normal(20, 2, n_samples).clip(18, 25).astype(int)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on features with better balance
        # Students with higher attendance, marks, and study hours are more likely to pass
        pass_probability = (
            df['attendance_percentage'] * 0.25 +
            df['assignment_marks'] * 0.2 +
            df['midterm_marks'] * 0.2 +
            df['final_exam_marks'] * 0.25 +
            df['study_hours_per_day'] * 8 +
            df['previous_semester_gpa'] * 15
        ) / 100
        
        # Add some randomness but ensure balanced classes
        pass_probability += np.random.normal(0, 0.15, n_samples)
        
        # Create binary outcome (Pass/Fail) with better balance
        # Use a threshold that ensures roughly 60-70% pass rate
        threshold = np.percentile(pass_probability, 35)  # This ensures ~65% pass rate
        df['result'] = (pass_probability > threshold).astype(int)
        
        # Ensure we have both classes
        if df['result'].nunique() < 2:
            # If we don't have both classes, adjust the threshold
            if df['result'].sum() == 0:  # All fails
                threshold = np.percentile(pass_probability, 50)
            else:  # All passes
                threshold = np.percentile(pass_probability, 70)
            df['result'] = (pass_probability > threshold).astype(int)
        
        # Create grade categories
        grade_conditions = [
            (df['final_exam_marks'] >= 90),
            (df['final_exam_marks'] >= 80),
            (df['final_exam_marks'] >= 70),
            (df['final_exam_marks'] >= 60),
            (df['final_exam_marks'] >= 50),
            (df['final_exam_marks'] < 50)
        ]
        grade_choices = ['A+', 'A', 'B+', 'B', 'C', 'F']
        df['grade'] = np.select(grade_conditions, grade_choices, default='F')
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Select features for prediction
        feature_columns = [
            'attendance_percentage', 'assignment_marks', 'midterm_marks',
            'final_exam_marks', 'study_hours_per_day', 'previous_semester_gpa',
            'extracurricular_activities', 'family_income', 'age'
        ]
        
        # Encode categorical variables
        df['gender_encoded'] = self.label_encoder.fit_transform(df['gender'])
        df['parent_education_encoded'] = self.label_encoder.fit_transform(df['parent_education'])
        
        feature_columns.extend(['gender_encoded', 'parent_education_encoded'])
        
        X = df[feature_columns]
        y_result = df['result']
        y_grade = df['grade']
        
        # Verify we have both classes before splitting
        print(f"Class distribution in target: {y_result.value_counts()}")
        if y_result.nunique() < 2:
            raise ValueError("Target variable must contain at least 2 classes for classification")
        
        # Split data with stratification only if we have sufficient samples per class
        min_samples_per_class = 10
        if y_result.value_counts().min() >= min_samples_per_class:
            X_train, X_test, y_train_result, y_test_result, y_train_grade, y_test_grade = train_test_split(
                X, y_result, y_grade, test_size=0.2, random_state=42, stratify=y_result
            )
        else:
            print("Warning: Insufficient samples per class for stratification, using random split")
            X_train, X_test, y_train_result, y_test_result, y_train_grade, y_test_grade = train_test_split(
                X, y_result, y_grade, test_size=0.2, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train_result, y_test_result, y_train_grade, y_test_grade
    
    def define_models(self):
        """Define all ML models to test"""
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
        }
    
    def train_models(self, X_train, y_train):
        """Train all models and find the best one"""
        print("Training multiple ML models...")
        results = {}
        
        for name, model_info in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Check if we have enough samples for cross-validation
                if len(y_train) < 10:
                    print(f"Warning: Insufficient samples for {name}, using simple train/validation split")
                    # Use simple train/validation split instead of cross-validation
                    from sklearn.model_selection import train_test_split
                    X_train_simple, X_val, y_train_simple, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42
                    )
                    
                    # Train with default parameters
                    model = model_info['model']
                    model.fit(X_train_simple, y_train_simple)
                    val_score = model.score(X_val, y_val)
                    
                    results[name] = {
                        'model': model,
                        'best_params': model.get_params(),
                        'best_score': val_score,
                        'cv_scores': None
                    }
                    
                    print(f"Validation score for {name}: {val_score:.4f}")
                    
                else:
                    # Grid search for hyperparameter tuning
                    grid_search = GridSearchCV(
                        model_info['model'],
                        model_info['params'],
                        cv=min(5, len(y_train) // 2),  # Adjust CV folds based on data size
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    # Store results
                    results[name] = {
                        'model': grid_search.best_estimator_,
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_,
                        'cv_scores': grid_search.cv_results_
                    }
                    
                    print(f"Best {name} score: {grid_search.best_score_:.4f}")
                    print(f"Best parameters: {grid_search.best_params_}")
                
                # Update best model
                if results[name]['best_score'] > self.best_score:
                    self.best_score = results[name]['best_score']
                    self.best_model = results[name]['model']
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                print(f"Skipping {name} and continuing with other models...")
                continue
        
        return results
    
    def evaluate_models(self, results, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        evaluation_results = {}
        
        for name, result in results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'model': model
            }
            
            print(f"\n{name.upper()}:")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Cross-validation Score: {result['best_score']:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Fail', 'Pass']))
        
        return evaluation_results
    
    def save_models(self, results, evaluation_results):
        """Save all trained models and metadata"""
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Save best model
        with open('model/best_predictor.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save all models
        with open('model/all_models.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Save evaluation results
        with open('model/evaluation_results.pkl', 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        # Save scaler and encoder
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open('model/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save model metadata
        model_info = {
            'best_model': type(self.best_model).__name__,
            'best_score': self.best_score,
            'feature_columns': [
                'attendance_percentage', 'assignment_marks', 'midterm_marks',
                'final_exam_marks', 'study_hours_per_day', 'previous_semester_gpa',
                'extracurricular_activities', 'family_income', 'age',
                'gender_encoded', 'parent_education_encoded'
            ],
            'model_comparison': {
                name: {
                    'accuracy': eval_results['accuracy'],
                    'cv_score': results[name]['best_score']
                }
                for name, eval_results in evaluation_results.items()
            }
        }
        
        with open('model/model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"\nModels saved successfully!")
        print(f"Best model: {type(self.best_model).__name__}")
        print(f"Best cross-validation score: {self.best_score:.4f}")

def main():
    """Main training function"""
    print("Student Result Predictor - Model Training")
    print("="*50)
    
    # Initialize predictor
    predictor = StudentResultPredictor()
    
    # Generate sample data
    print("Generating sample training data...")
    df = predictor.generate_sample_data(n_samples=2000)
    print(f"Generated {len(df)} student records")
    
    # Prepare data
    print("Preparing data for training...")
    X_train, X_test, y_train, y_test, y_train_grade, y_test_grade = predictor.prepare_data(df)
    
    # Define models
    predictor.define_models()
    
    # Train models
    results = predictor.train_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = predictor.evaluate_models(results, X_test, y_test)
    
    # Save models
    predictor.save_models(results, evaluation_results)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("You can now run the Flask application with: python app.py")

if __name__ == "__main__":
    main() 