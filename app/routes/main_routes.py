from flask import Blueprint, render_template, flash, redirect, url_for, request
from app.services.excel_service import excel_handler
from app.services.predictor_service import predictor

main_bp = Blueprint('main', __name__)

# Add predict route to main blueprint
@main_bp.route('/predict', methods=['GET', 'POST'])
def predict():
	# TODO: Add prediction logic here or import from service
	return render_template('form.html')
# Main routes for the app

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

@main_bp.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		# TODO: Implement file upload logic
		flash('Upload functionality coming soon!', 'info')
		return redirect(url_for('main.data'))
	return render_template('upload.html')

@main_bp.route('/chatbot')
def chatbot_route():
	# TODO: Implement chatbot functionality
	return render_template('chatbot.html')

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
# Main routes will be moved here
