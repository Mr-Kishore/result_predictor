# Excel business logic
from utils.excel_handler import ExcelHandler
excel_handler = ExcelHandler()

# Expose the detailed statistics methods
def get_detailed_pass_rate_stats():
    return excel_handler.get_detailed_pass_rate_stats()

def get_detailed_attendance_stats():
    return excel_handler.get_detailed_attendance_stats()

def get_detailed_marks_stats():
    return excel_handler.get_detailed_marks_stats()
