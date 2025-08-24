# API routes for the app
from flask import Blueprint, request, jsonify
from app.services.excel_service import excel_handler
from app.services.predictor_service import predictor

api_bp = Blueprint('api', __name__)

@api_bp.route('/api/predict', methods=['POST'])
def api_predict():
	try:
		data = request.get_json()
		if not data or not isinstance(data, dict):
			return jsonify({'success': False, 'message': 'Invalid request: JSON body required.'}), 400

		student_data = data.get('student_data')
		model_name = data.get('model', 'best')

		# Strict validation: required fields, types, and value ranges
		required_fields = [
			'attendance_percentage', 'assignment_marks', 'midterm_marks',
			'final_exam_marks', 'study_hours_per_day', 'previous_semester_gpa'
		]
		errors = []
		if not student_data or not isinstance(student_data, dict):
			errors.append('student_data must be a dictionary.')
		else:
			for field in required_fields:
				if field not in student_data:
					errors.append(f'Missing required field: {field}')
				else:
					value = student_data[field]
					try:
						value = float(value)
					except (ValueError, TypeError):
						errors.append(f'{field} must be a number.')
						continue
					if field == 'attendance_percentage' and not (0 <= value <= 100):
						errors.append('attendance_percentage must be between 0 and 100.')
					if field in ['assignment_marks', 'midterm_marks', 'final_exam_marks'] and not (0 <= value <= 100):
						errors.append(f'{field} must be between 0 and 100.')
					if field == 'study_hours_per_day' and not (0 <= value <= 24):
						errors.append('study_hours_per_day must be between 0 and 24.')
					if field == 'previous_semester_gpa' and not (0 <= value <= 4):
						errors.append('previous_semester_gpa must be between 0 and 4.')

		if errors:
			return jsonify({'success': False, 'message': 'Validation failed', 'errors': errors}), 400

		# Additional validation from predictor
		validation = predictor.validate_input(student_data)
		if not validation['valid']:
			return jsonify({'success': False, 'message': 'Validation failed', 'errors': validation['errors']}), 400

		prediction_result = predictor.predict_result(student_data, model_name)
		if prediction_result['success']:
			insights = predictor.get_prediction_insights(student_data, prediction_result)
			student_data.update({
				'predicted_result': prediction_result['predicted_result'],
				'predicted_grade': prediction_result['predicted_grade'],
				'confidence_score': prediction_result['confidence_score'],
				'model_used': prediction_result['model_used']
			})
			save_result = excel_handler.add_manual_entry(student_data)
			return jsonify({'success': True, 'prediction': prediction_result, 'insights': insights, 'save_result': save_result})
		else:
			return jsonify({'success': False, 'message': prediction_result['message']}), 400
	except Exception as e:
		return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@api_bp.route('/api/stats')
def api_stats():
	try:
		stats = excel_handler.get_statistics()
		model_info = predictor.get_model_comparison()
		stats['recent_data'] = excel_handler.get_recent_data(50)
		return jsonify({'success': True, 'stats': stats, 'model_info': model_info})
	except Exception as e:
		return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@api_bp.route('/api/feature-importance')
def api_feature_importance():
	try:
		model_name = request.args.get('model', 'best')
		result = predictor.get_feature_importance(model_name)
		return jsonify(result)
	except Exception as e:
		return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@api_bp.route('/api/detailed-stats/pass-rate')
def api_detailed_pass_rate():
	try:
		from app.services.excel_service import get_detailed_pass_rate_stats
		stats = get_detailed_pass_rate_stats()
		return jsonify({'success': True, 'data': stats})
	except Exception as e:
		return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@api_bp.route('/api/detailed-stats/attendance')
def api_detailed_attendance():
	try:
		from app.services.excel_service import get_detailed_attendance_stats
		stats = get_detailed_attendance_stats()
		return jsonify({'success': True, 'data': stats})
	except Exception as e:
		return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@api_bp.route('/api/detailed-stats/marks')
def api_detailed_marks():
	try:
		from app.services.excel_service import get_detailed_marks_stats
		stats = get_detailed_marks_stats()
		return jsonify({'success': True, 'data': stats})
	except Exception as e:
		return jsonify({'success': False, 'message': f'Error: {str(e)}'})
