from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import json
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime

# Import our custom modules
from utils.excel_handler import ExcelHandler
from utils.predictor import StudentPredictor
from chatbot.chatbot import StudentChatbot

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Initialize our modules
excel_handler = ExcelHandler()
predictor = StudentPredictor()
chatbot = StudentChatbot()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page"""
    # Get statistics for dashboard
    stats = excel_handler.get_statistics()
    model_info = predictor.get_model_comparison()
    
    return render_template('index.html', 
                         stats=stats, 
                         model_info=model_info,
                         available_models=list(predictor.models.keys()))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            # Get form data
            student_data = {
                'student_id': request.form.get('student_id', ''),
                'name': request.form.get('name', ''),
                'gender': request.form.get('gender', ''),
                'age': request.form.get('age', ''),
                'attendance_percentage': request.form.get('attendance_percentage', ''),
                'assignment_marks': request.form.get('assignment_marks', ''),
                'midterm_marks': request.form.get('midterm_marks', ''),
                'final_exam_marks': request.form.get('final_exam_marks', ''),
                'study_hours_per_day': request.form.get('study_hours_per_day', ''),
                'previous_semester_gpa': request.form.get('previous_semester_gpa', ''),
                'extracurricular_activities': request.form.get('extracurricular_activities', '')
            }
            
            # Validate input
            validation = predictor.validate_input(student_data)
            if not validation['valid']:
                flash('Please fix the following errors: ' + ', '.join(validation['errors']), 'error')
                return render_template('form.html', student_data=student_data, errors=validation['errors'])
            
            # Get selected model
            selected_model = request.form.get('model', 'best')
            
            # Make prediction
            prediction_result = predictor.predict_result(student_data, selected_model)
            
            if prediction_result['success']:
                # Get insights
                insights = predictor.get_prediction_insights(student_data, prediction_result)
                
                # Add prediction data to student_data for saving
                student_data.update({
                    'predicted_result': prediction_result['predicted_result'],
                    'predicted_grade': prediction_result['predicted_grade'],
                    'confidence_score': prediction_result['confidence_score'],
                    'model_used': prediction_result['model_used']
                })
                
                # Save to Excel
                save_result = excel_handler.add_manual_entry(student_data)
                
                return render_template('result.html', 
                                     prediction=prediction_result,
                                     student_data=student_data,
                                     insights=insights,
                                     save_result=save_result)
            else:
                flash(f'Prediction failed: {prediction_result["message"]}', 'error')
                return render_template('form.html', student_data=student_data)
                
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            return render_template('form.html', student_data={})
    
    return render_template('form.html', student_data={})

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """File upload page"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                # Secure the filename
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the uploaded file
                result = excel_handler.process_uploaded_file(filepath)
                
                if result['success']:
                    flash(result['message'], 'success')
                    
                    # If there are predictions to make, do batch prediction
                    if result['new_records'] > 0:
                        # Load the updated data to get new records
                        df = excel_handler.load_master_data()
                        new_records = df.tail(result['new_records'])
                        
                        # Make predictions for new records
                        predictions_made = 0
                        for idx, row in new_records.iterrows():
                            if pd.isna(row.get('predicted_result')):
                                student_data = row.to_dict()
                                prediction = predictor.predict_result(student_data)
                                if prediction['success']:
                                    df.at[idx, 'predicted_result'] = prediction['predicted_result']
                                    df.at[idx, 'predicted_grade'] = prediction['predicted_grade']
                                    df.at[idx, 'confidence_score'] = prediction['confidence_score']
                                    df.at[idx, 'model_used'] = prediction['model_used']
                                    predictions_made += 1
                        
                        # Save updated predictions
                        if predictions_made > 0:
                            excel_handler.save_master_data(df)
                            flash(f'Made predictions for {predictions_made} new records', 'success')
                    
                    return redirect(url_for('index'))
                else:
                    flash(result['message'], 'error')
                    return redirect(request.url)
            else:
                flash('Invalid file type. Please upload Excel files (.xlsx, .xls)', 'error')
                return redirect(request.url)
                
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_route():
    """Chatbot interface"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            user_message = data.get('message', '')
            
            if user_message.strip():
                # Get chatbot response
                response = chatbot.get_response(user_message)
                
                # Get suggestions
                suggestions = chatbot.get_suggestions(user_message)
                
                return jsonify({
                    'success': True,
                    'response': response['response'],
                    'confidence': response['confidence'],
                    'suggestions': suggestions
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Please enter a message'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            })
    
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chatbot"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if user_message.strip():
            response = chatbot.get_response(user_message)
            return jsonify({
                'success': True,
                'response': response['response'],
                'confidence': response['confidence']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Empty message'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        student_data = data.get('student_data', {})
        model_name = data.get('model', 'best')
        
        # Validate input
        validation = predictor.validate_input(student_data)
        if not validation['valid']:
            return jsonify({
                'success': False,
                'message': 'Validation failed',
                'errors': validation['errors']
            })
        
        # Make prediction
        prediction_result = predictor.predict_result(student_data, model_name)
        
        if prediction_result['success']:
            # Get insights
            insights = predictor.get_prediction_insights(student_data, prediction_result)
            
            # Add prediction data for saving
            student_data.update({
                'predicted_result': prediction_result['predicted_result'],
                'predicted_grade': prediction_result['predicted_grade'],
                'confidence_score': prediction_result['confidence_score'],
                'model_used': prediction_result['model_used']
            })
            
            # Save to Excel
            save_result = excel_handler.add_manual_entry(student_data)
            
            return jsonify({
                'success': True,
                'prediction': prediction_result,
                'insights': insights,
                'save_result': save_result
            })
        else:
            return jsonify({
                'success': False,
                'message': prediction_result['message']
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded'
            })
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            })
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = excel_handler.process_uploaded_file(filepath)
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid file type'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    try:
        stats = excel_handler.get_statistics()
        model_info = predictor.get_model_comparison()
        
        # Add recent data for data management page
        stats['recent_data'] = excel_handler.get_recent_data(50)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'model_info': model_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/export', methods=['POST'])
def api_export():
    """API endpoint for data export"""
    try:
        data = request.get_json()
        export_format = data.get('format', 'excel')
        filename = data.get('filename', None)
        
        result = excel_handler.export_data(export_format, filename)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/api/feature-importance')
def api_feature_importance():
    """API endpoint for feature importance"""
    try:
        model_name = request.args.get('model', 'best')
        result = predictor.get_feature_importance(model_name)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/models')
def models():
    """Models comparison page"""
    model_info = predictor.get_model_comparison()
    feature_importance = predictor.get_feature_importance('best')
    
    return render_template('models.html', 
                         model_info=model_info,
                         feature_importance=feature_importance)

@app.route('/data')
def data():
    """Data management page"""
    stats = excel_handler.get_statistics()
    return render_template('data.html', stats=stats)

@app.route('/search', methods=['GET', 'POST'])
def search_student():
    """Student search page"""
    if request.method == 'POST':
        roll_number = request.form.get('roll_number', '').strip()
        if roll_number:
            # Search for student in the database
            student_data = excel_handler.search_student_by_id(roll_number)
            if student_data:
                return render_template('student_detail.html', student=student_data)
            else:
                flash(f'No student found with Roll Number: {roll_number}', 'warning')
        else:
            flash('Please enter a Roll Number', 'error')
    
    return render_template('search.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("Starting Student Result Predictor...")
    print("="*50)
    print("Available models:", list(predictor.models.keys()))
    print("Best model:", type(predictor.best_model).__name__ if predictor.best_model else "None")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 