from flask import Blueprint, request, jsonify
from ..managers.excel_manager import excel_handler
from ..managers.predictor_manager import predictor

api_bp = Blueprint('api', __name__)

@api_bp.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        student_data = data.get('student_data', {})
        model_name = data.get('model', 'best')
        validation = predictor.validate_input(student_data)
        if not validation['valid']:
            return jsonify({'success': False, 'message': 'Validation failed', 'errors': validation['errors']})
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
            return jsonify({'success': False, 'message': prediction_result['message']})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

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
        stats = excel_handler.get_detailed_pass_rate_stats()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@api_bp.route('/api/detailed-stats/attendance')
def api_detailed_attendance():
    try:
        stats = excel_handler.get_detailed_attendance_stats()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@api_bp.route('/api/detailed-stats/marks')
def api_detailed_marks():
    try:
        stats = excel_handler.get_detailed_marks_stats()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})
