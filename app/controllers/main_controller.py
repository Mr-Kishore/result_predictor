from flask import Blueprint, render_template, flash, redirect, url_for, request
from ..managers.excel_manager import excel_handler
from ..managers.predictor_manager import predictor

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    stats = excel_handler.get_statistics()
    model_info = predictor.get_model_comparison()
    return render_template('index.html', stats=stats, model_info=model_info, available_models=list(predictor.models.keys()))

@main_bp.route('/models')
def models():
    model_info = predictor.get_model_comparison()
    feature_importance = predictor.get_feature_importance('best')
    return render_template('models.html', model_info=model_info, feature_importance=feature_importance)

@main_bp.route('/data')
def data():
    stats = excel_handler.get_statistics()
    return render_template('data.html', stats=stats)

@main_bp.route('/search', methods=['GET', 'POST'])
def search_student():
    if request.method == 'POST':
        roll_number = request.form.get('roll_number', '').strip()
        if roll_number:
            student_data = excel_handler.search_student_by_id(roll_number)
            if student_data:
                return render_template('student_detail.html', student=student_data)
            else:
                flash(f'No student found with Roll Number: {roll_number}', 'warning')
        else:
            flash('Please enter a Roll Number', 'error')
    return render_template('search.html')

@main_bp.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@main_bp.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500
