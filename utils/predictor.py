import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class StudentPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.label_encoder = None
        self.model_info = None
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load all trained models and metadata"""
        try:
            model_dir = 'model'
            
            # Load best model
            if os.path.exists(f'{model_dir}/best_predictor.pkl'):
                with open(f'{model_dir}/best_predictor.pkl', 'rb') as f:
                    self.best_model = pickle.load(f)
            
            # Load all models
            if os.path.exists(f'{model_dir}/all_models.pkl'):
                with open(f'{model_dir}/all_models.pkl', 'rb') as f:
                    all_models_data = pickle.load(f)
                    self.models = {name: data['model'] for name, data in all_models_data.items()}
            
            # Load scaler
            if os.path.exists(f'{model_dir}/scaler.pkl'):
                with open(f'{model_dir}/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load label encoder
            if os.path.exists(f'{model_dir}/label_encoder.pkl'):
                with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
                    self.label_encoder = pickle.load(f)
            
            # Load model info
            if os.path.exists(f'{model_dir}/model_info.pkl'):
                with open(f'{model_dir}/model_info.pkl', 'rb') as f:
                    self.model_info = pickle.load(f)
                    self.feature_columns = self.model_info.get('feature_columns', [])
            
            print("Models loaded successfully!")
            if self.best_model:
                print(f"Best model: {type(self.best_model).__name__}")
            print(f"Available models: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please run train_model.py first to train the models.")
    
    def preprocess_input(self, student_data):
        """Preprocess input data for prediction"""
        try:
            # Create feature vector
            features = []
            
            # Numeric features
            numeric_features = [
                'attendance_percentage', 'assignment_marks', 'midterm_marks',
                'final_exam_marks', 'study_hours_per_day', 'previous_semester_gpa',
                'extracurricular_activities', 'age'
            ]
            
            for feature in numeric_features:
                if feature in student_data:
                    value = student_data[feature]
                    if pd.isna(value) or value == '':
                        # Use median values for missing data
                        default_values = {
                            'attendance_percentage': 85,
                            'assignment_marks': 75,
                            'midterm_marks': 70,
                            'final_exam_marks': 65,
                            'study_hours_per_day': 3,
                            'previous_semester_gpa': 3.0,
                            'extracurricular_activities': 1,
                            'age': 20
                        }
                        features.append(default_values[feature])
                    else:
                        features.append(float(value))
                else:
                    # Use default values for missing features
                    default_values = {
                        'attendance_percentage': 85,
                        'assignment_marks': 75,
                        'midterm_marks': 70,
                        'final_exam_marks': 65,
                        'study_hours_per_day': 3,
                        'previous_semester_gpa': 3.0,
                        'extracurricular_activities': 1,
                        'age': 20
                    }
                    features.append(default_values[feature])
            
            # Add default values for removed fields (family_income and parent_education)
            features.append(50000)  # Default family income
            features.append(0)      # Default parent education (encoded)
            
            # Categorical features
            if 'gender' in student_data and student_data['gender']:
                gender_encoded = self.label_encoder.transform([student_data['gender']])[0]
            else:
                gender_encoded = 0  # Default to first category
            features.append(gender_encoded)
            
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler:
                features_scaled = self.scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            return features_scaled
            
        except Exception as e:
            print(f"Error preprocessing input: {e}")
            return None
    
    def predict_result(self, student_data, model_name='best'):
        """Predict student result using specified model"""
        try:
            # Preprocess input
            features = self.preprocess_input(student_data)
            if features is None:
                return {
                    'success': False,
                    'message': 'Error preprocessing input data'
                }
            
            # Select model
            if model_name == 'best' and self.best_model:
                model = self.best_model
                model_type = type(self.best_model).__name__
            elif model_name in self.models:
                model = self.models[model_name]
                model_type = type(model).__name__
            else:
                return {
                    'success': False,
                    'message': f'Model "{model_name}" not found. Available models: {list(self.models.keys())}'
                }
            
            # Make prediction
            prediction = model.predict(features)[0]
            prediction_proba = model.predict_proba(features)[0]
            
            # Get confidence score
            confidence = max(prediction_proba) * 100
            
            # Determine grade based on final exam marks
            final_exam_marks = float(student_data.get('final_exam_marks', 65))
            grade = self.calculate_grade(final_exam_marks)
            
            # Prepare result
            result = {
                'success': True,
                'predicted_result': int(prediction),  # 0 for Fail, 1 for Pass
                'result_text': 'Pass' if prediction == 1 else 'Fail',
                'predicted_grade': grade,
                'confidence_score': round(confidence, 2),
                'model_used': model_type,
                'pass_probability': round(prediction_proba[1] * 100, 2),
                'fail_probability': round(prediction_proba[0] * 100, 2)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error making prediction: {str(e)}'
            }
    
    def predict_batch(self, student_data_list, model_name='best'):
        """Predict results for multiple students"""
        try:
            results = []
            
            for i, student_data in enumerate(student_data_list):
                result = self.predict_result(student_data, model_name)
                result['student_index'] = i
                results.append(result)
            
            return {
                'success': True,
                'predictions': results,
                'total_students': len(student_data_list)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error in batch prediction: {str(e)}'
            }
    
    def calculate_grade(self, final_exam_marks):
        """Calculate grade based on final exam marks"""
        if final_exam_marks >= 90:
            return 'A+'
        elif final_exam_marks >= 80:
            return 'A'
        elif final_exam_marks >= 70:
            return 'B+'
        elif final_exam_marks >= 60:
            return 'B'
        elif final_exam_marks >= 50:
            return 'C'
        else:
            return 'F'
    
    def get_model_comparison(self):
        """Get comparison of all available models"""
        if not self.model_info:
            return {
                'success': False,
                'message': 'Model information not available'
            }
        
        return {
            'success': True,
            'best_model': self.model_info.get('best_model', 'Unknown'),
            'best_score': self.model_info.get('best_score', 0),
            'available_models': list(self.models.keys()),
            'model_comparison': self.model_info.get('model_comparison', {}),
            'feature_columns': self.feature_columns
        }
    
    def get_feature_importance(self, model_name='best'):
        """Get feature importance for tree-based models"""
        try:
            if model_name == 'best' and self.best_model:
                model = self.best_model
            elif model_name in self.models:
                model = self.models[model_name]
            else:
                return {
                    'success': False,
                    'message': f'Model "{model_name}" not found'
                }
            
            # Check if model supports feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                return {
                    'success': False,
                    'message': 'This model does not support feature importance'
                }
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_columns):
                feature_importance[feature] = float(importance[i])
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'success': True,
                'feature_importance': dict(sorted_features),
                'top_features': [feature for feature, _ in sorted_features[:5]]
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error getting feature importance: {str(e)}'
            }
    
    def validate_input(self, student_data):
        """Validate input data"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = [
            'attendance_percentage', 'assignment_marks', 'midterm_marks',
            'final_exam_marks', 'study_hours_per_day', 'previous_semester_gpa'
        ]
        
        for field in required_fields:
            if field not in student_data or student_data[field] == '':
                errors.append(f'Missing required field: {field}')
            else:
                try:
                    value = float(student_data[field])
                    # Check ranges
                    if field == 'attendance_percentage' and (value < 0 or value > 100):
                        warnings.append(f'{field} should be between 0 and 100')
                    elif field in ['assignment_marks', 'midterm_marks', 'final_exam_marks'] and (value < 0 or value > 100):
                        warnings.append(f'{field} should be between 0 and 100')
                    elif field == 'study_hours_per_day' and (value < 0 or value > 24):
                        warnings.append(f'{field} should be between 0 and 24')
                    elif field == 'previous_semester_gpa' and (value < 0 or value > 4):
                        warnings.append(f'{field} should be between 0 and 4')
                except ValueError:
                    errors.append(f'{field} should be a valid number')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_prediction_insights(self, student_data, prediction_result):
        """Get insights about the prediction"""
        try:
            insights = []
            
            # Analyze attendance
            attendance = float(student_data.get('attendance_percentage', 0))
            if attendance < 75:
                insights.append("Low attendance may negatively impact your performance")
            elif attendance > 90:
                insights.append("Excellent attendance! This will help your success")
            
            # Analyze study hours
            study_hours = float(student_data.get('study_hours_per_day', 0))
            if study_hours < 2:
                insights.append("Consider increasing study hours for better performance")
            elif study_hours > 6:
                insights.append("Good study habits! Maintain this consistency")
            
            # Analyze marks progression
            assignment_marks = float(student_data.get('assignment_marks', 0))
            midterm_marks = float(student_data.get('midterm_marks', 0))
            final_marks = float(student_data.get('final_exam_marks', 0))
            
            if final_marks < midterm_marks:
                insights.append("Final exam performance dropped from midterm - consider additional preparation")
            elif final_marks > midterm_marks:
                insights.append("Great improvement from midterm to final exam!")
            
            # Overall performance
            avg_marks = (assignment_marks + midterm_marks + final_marks) / 3
            if avg_marks < 60:
                insights.append("Overall marks are below average - consider seeking academic support")
            elif avg_marks > 80:
                insights.append("Excellent overall performance! Keep up the good work")
            
            return {
                'success': True,
                'insights': insights,
                'confidence_level': 'High' if prediction_result['confidence_score'] > 80 else 'Medium' if prediction_result['confidence_score'] > 60 else 'Low'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error generating insights: {str(e)}'
            } 