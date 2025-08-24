import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ExcelHandler:
    def __init__(self, master_file='data/student_records.xlsx'):
        self.master_file = master_file
        self.ensure_data_directory()
        self.ensure_master_file()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs('data', exist_ok=True)
    
    def ensure_master_file(self):
        """Create master Excel file with proper structure if it doesn't exist"""
        if not os.path.exists(self.master_file):
            # Create empty DataFrame with proper columns
            columns = [
                'student_id', 'name', 'gender', 'age', 'attendance_percentage',
                'assignment_marks', 'midterm_marks', 'final_exam_marks',
                'study_hours_per_day', 'previous_semester_gpa',
                'extracurricular_activities', 'family_income', 'parent_education',
                'predicted_result', 'predicted_grade', 'confidence_score',
                'model_used', 'timestamp', 'entry_method'
            ]
            
            df = pd.DataFrame(columns=columns)
            df.to_excel(self.master_file, index=False)
            print(f"Created new master file: {self.master_file}")
    
    def load_master_data(self):
        """Load existing master data"""
        try:
            if os.path.exists(self.master_file):
                df = pd.read_excel(self.master_file)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading master data: {e}")
            return pd.DataFrame()
    
    def save_master_data(self, df):
        """Save data to master Excel file"""
        try:
            df.to_excel(self.master_file, index=False)
            return True
        except Exception as e:
            print(f"Error saving master data: {e}")
            return False
    
    def add_manual_entry(self, student_data):
        """Add a single manual entry to master file"""
        try:
            # Load existing data
            df = self.load_master_data()
            
            # Add timestamp and entry method
            student_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            student_data['entry_method'] = 'manual'
            
            # Convert to DataFrame
            new_entry = pd.DataFrame([student_data])
            
            # Check for duplicates based on student_id
            if 'student_id' in student_data and student_data['student_id']:
                if not df.empty and 'student_id' in df.columns:
                    existing_ids = df['student_id'].dropna().astype(str)
                    new_id = str(student_data['student_id'])
                    
                    if new_id in existing_ids.values:
                        return {
                            'success': False,
                            'message': f'Student ID {new_id} already exists in records.',
                            'duplicate': True
                        }
            
            # Append new entry
            df = pd.concat([df, new_entry], ignore_index=True)
            
            # Save updated data
            if self.save_master_data(df):
                return {
                    'success': True,
                    'message': 'Student record added successfully.',
                    'total_records': len(df)
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to save student record.'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error adding manual entry: {str(e)}'
            }
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded Excel file and merge with master data"""
        try:
            # Read uploaded file
            uploaded_df = pd.read_excel(uploaded_file)
            
            # Load existing master data
            master_df = self.load_master_data()
            
            # Standardize column names
            uploaded_df = self.standardize_columns(uploaded_df)
            
            # Validate required columns
            required_columns = [
                'student_id', 'attendance_percentage', 'assignment_marks',
                'midterm_marks', 'final_exam_marks', 'study_hours_per_day',
                'previous_semester_gpa'
            ]
            
            missing_columns = [col for col in required_columns if col not in uploaded_df.columns]
            if missing_columns:
                return {
                    'success': False,
                    'message': f'Missing required columns: {", ".join(missing_columns)}'
                }
            
            # Add metadata columns if not present
            if 'timestamp' not in uploaded_df.columns:
                uploaded_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if 'entry_method' not in uploaded_df.columns:
                uploaded_df['entry_method'] = 'upload'
            
            # Check for duplicates
            duplicate_info = self.check_duplicates(master_df, uploaded_df)
            
            # Filter out duplicates
            new_records = uploaded_df[~duplicate_info['duplicate_mask']]
            
            # Merge with master data
            if not master_df.empty:
                combined_df = pd.concat([master_df, new_records], ignore_index=True)
            else:
                combined_df = new_records
            
            # Save updated master data
            if self.save_master_data(combined_df):
                return {
                    'success': True,
                    'message': f'Successfully processed {len(new_records)} new records. {len(duplicate_info["duplicates"])} duplicates found and skipped.',
                    'total_records': len(combined_df),
                    'new_records': len(new_records),
                    'duplicates': len(duplicate_info['duplicates']),
                    'duplicate_details': duplicate_info['duplicates']
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to save updated master data.'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing uploaded file: {str(e)}'
            }
    
    def standardize_columns(self, df):
        """Standardize column names to match master file format"""
        column_mapping = {
            'Student ID': 'student_id',
            'Student_ID': 'student_id',
            'studentid': 'student_id',
            'Name': 'name',
            'Student Name': 'name',
            'Gender': 'gender',
            'Age': 'age',
            'Attendance': 'attendance_percentage',
            'Attendance %': 'attendance_percentage',
            'Attendance_Percentage': 'attendance_percentage',
            'Assignment Marks': 'assignment_marks',
            'Assignment_Marks': 'assignment_marks',
            'Midterm Marks': 'midterm_marks',
            'Midterm_Marks': 'midterm_marks',
            'Final Exam Marks': 'final_exam_marks',
            'Final_Exam_Marks': 'final_exam_marks',
            'Study Hours': 'study_hours_per_day',
            'Study_Hours': 'study_hours_per_day',
            'GPA': 'previous_semester_gpa',
            'Previous GPA': 'previous_semester_gpa',
            'Previous_Semester_GPA': 'previous_semester_gpa',
            'Extracurricular': 'extracurricular_activities',
            'Extracurricular_Activities': 'extracurricular_activities',
            'Family Income': 'family_income',
            'Family_Income': 'family_income',
            'Parent Education': 'parent_education',
            'Parent_Education': 'parent_education'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        return df
    
    def check_duplicates(self, master_df, new_df):
        """Check for duplicate records between master and new data"""
        if master_df.empty:
            return {
                'duplicate_mask': pd.Series([False] * len(new_df)),
                'duplicates': []
            }
        
        duplicates = []
        duplicate_mask = pd.Series([False] * len(new_df))
        
        for idx, new_row in new_df.iterrows():
            is_duplicate = False
            
            # Check by student_id if available
            if 'student_id' in new_row and pd.notna(new_row['student_id']):
                new_id = str(new_row['student_id'])
                if 'student_id' in master_df.columns:
                    existing_ids = master_df['student_id'].dropna().astype(str)
                    if new_id in existing_ids.values:
                        is_duplicate = True
                        duplicates.append({
                            'row_index': idx,
                            'student_id': new_id,
                            'reason': 'Student ID already exists'
                        })
            
            # Check by full row match if no student_id match
            if not is_duplicate:
                # Compare key columns
                key_columns = [
                    'attendance_percentage', 'assignment_marks', 'midterm_marks',
                    'final_exam_marks', 'study_hours_per_day', 'previous_semester_gpa'
                ]
                
                for _, master_row in master_df.iterrows():
                    match = True
                    for col in key_columns:
                        if col in new_row and col in master_row:
                            if pd.notna(new_row[col]) and pd.notna(master_row[col]):
                                if abs(float(new_row[col]) - float(master_row[col])) > 0.01:
                                    match = False
                                    break
                    
                    if match:
                        is_duplicate = True
                        duplicates.append({
                            'row_index': idx,
                            'student_id': new_row.get('student_id', 'Unknown'),
                            'reason': 'Full row match found'
                        })
                        break
            
            duplicate_mask[idx] = is_duplicate
        
        return {
            'duplicate_mask': duplicate_mask,
            'duplicates': duplicates
        }
    
    def get_statistics(self):
        """Get statistics about the master data"""
        try:
            df = self.load_master_data()
            
            if df.empty:
                return {
                    'total_records': 0,
                    'pass_rate': 0,
                    'average_attendance': 0,
                    'average_marks': 0
                }
            
            stats = {
                'total_records': len(df),
                'pass_rate': 0,
                'average_attendance': 0,
                'average_marks': 0,
                'recent_entries': 0
            }
            
            # Calculate pass rate
            if 'predicted_result' in df.columns:
                pass_count = df['predicted_result'].sum()
                stats['pass_rate'] = (pass_count / len(df)) * 100 if len(df) > 0 else 0
            
            # Calculate average attendance
            if 'attendance_percentage' in df.columns:
                stats['average_attendance'] = df['attendance_percentage'].mean()
            
            # Calculate average marks
            mark_columns = ['assignment_marks', 'midterm_marks', 'final_exam_marks']
            available_marks = [col for col in mark_columns if col in df.columns]
            if available_marks:
                stats['average_marks'] = df[available_marks].mean().mean()
            
            # Count recent entries (last 7 days)
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    recent_date = pd.Timestamp.now() - pd.Timedelta(days=7)
                    stats['recent_entries'] = len(df[df['timestamp'] >= recent_date])
                except:
                    stats['recent_entries'] = 0
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'total_records': 0,
                'pass_rate': 0,
                'average_attendance': 0,
                'average_marks': 0,
                'recent_entries': 0
            }

    def get_detailed_pass_rate_stats(self):
        """Get detailed pass/fail statistics with student details"""
        try:
            df = self.load_master_data()
            
            if df.empty or 'predicted_result' not in df.columns:
                return {
                    'passed_students': [],
                    'failed_students': [],
                    'pass_count': 0,
                    'fail_count': 0,
                    'pass_rate': 0
                }
            
            # Separate passed and failed students
            passed_df = df[df['predicted_result'] == 1]
            failed_df = df[df['predicted_result'] == 0]
            
            passed_students = []
            for _, student in passed_df.iterrows():
                passed_students.append({
                    'student_id': student.get('student_id', 'N/A'),
                    'name': student.get('name', 'N/A'),
                    'predicted_grade': student.get('predicted_grade', 'N/A'),
                    'confidence_score': student.get('confidence_score', 0)
                })
            
            failed_students = []
            for _, student in failed_df.iterrows():
                failed_students.append({
                    'student_id': student.get('student_id', 'N/A'),
                    'name': student.get('name', 'N/A'),
                    'predicted_grade': student.get('predicted_grade', 'N/A'),
                    'confidence_score': student.get('confidence_score', 0)
                })
            
            pass_count = len(passed_students)
            fail_count = len(failed_students)
            total = pass_count + fail_count
            pass_rate = (pass_count / total * 100) if total > 0 else 0
            
            return {
                'passed_students': passed_students,
                'failed_students': failed_students,
                'pass_count': pass_count,
                'fail_count': fail_count,
                'pass_rate': pass_rate
            }
            
        except Exception as e:
            print(f"Error getting detailed pass rate stats: {e}")
            return {
                'passed_students': [],
                'failed_students': [],
                'pass_count': 0,
                'fail_count': 0,
                'pass_rate': 0
            }

    def get_detailed_attendance_stats(self):
        """Get detailed attendance statistics with daily breakdown"""
        try:
            df = self.load_master_data()
            
            if df.empty or 'attendance_percentage' not in df.columns:
                return {
                    'average_attendance': 0,
                    'present_count': 0,
                    'absent_count': 0,
                    'attendance_ranges': {}
                }
            
            avg_attendance = df['attendance_percentage'].mean()
            
            # Calculate present/absent based on attendance threshold (75% as present)
            present_count = len(df[df['attendance_percentage'] >= 75])
            absent_count = len(df[df['attendance_percentage'] < 75])
            
            # Attendance ranges
            ranges = {
                '90-100%': len(df[df['attendance_percentage'] >= 90]),
                '80-89%': len(df[(df['attendance_percentage'] >= 80) & (df['attendance_percentage'] < 90)]),
                '70-79%': len(df[(df['attendance_percentage'] >= 70) & (df['attendance_percentage'] < 80)]),
                '60-69%': len(df[(df['attendance_percentage'] >= 60) & (df['attendance_percentage'] < 70)]),
                'Below 60%': len(df[df['attendance_percentage'] < 60])
            }
            
            return {
                'average_attendance': avg_attendance,
                'present_count': present_count,
                'absent_count': absent_count,
                'attendance_ranges': ranges,
                'total_students': len(df)
            }
            
        except Exception as e:
            print(f"Error getting detailed attendance stats: {e}")
            return {
                'average_attendance': 0,
                'present_count': 0,
                'absent_count': 0,
                'attendance_ranges': {},
                'total_students': 0
            }

    def get_detailed_marks_stats(self):
        """Get detailed marks statistics with grade distribution"""
        try:
            df = self.load_master_data()
            
            mark_columns = ['assignment_marks', 'midterm_marks', 'final_exam_marks']
            available_marks = [col for col in mark_columns if col in df.columns]
            
            if df.empty or not available_marks:
                return {
                    'average_marks': 0,
                    'grade_distribution': {},
                    'subject_averages': {}
                }
            
            # Calculate overall average
            df['overall_average'] = df[available_marks].mean(axis=1)
            avg_marks = df['overall_average'].mean()
            
            # Grade distribution
            grade_distribution = {
                'A+ (90-100)': len(df[df['overall_average'] >= 90]),
                'A (80-89)': len(df[(df['overall_average'] >= 80) & (df['overall_average'] < 90)]),
                'B+ (70-79)': len(df[(df['overall_average'] >= 70) & (df['overall_average'] < 80)]),
                'B (60-69)': len(df[(df['overall_average'] >= 60) & (df['overall_average'] < 70)]),
                'C (50-59)': len(df[(df['overall_average'] >= 50) & (df['overall_average'] < 60)]),
                'F (Below 50)': len(df[df['overall_average'] < 50])
            }
            
            # Subject averages
            subject_averages = {}
            for col in available_marks:
                subject_name = col.replace('_marks', '').replace('_', ' ').title()
                subject_averages[subject_name] = df[col].mean()
            
            return {
                'average_marks': avg_marks,
                'grade_distribution': grade_distribution,
                'subject_averages': subject_averages,
                'total_students': len(df)
            }
            
        except Exception as e:
            print(f"Error getting detailed marks stats: {e}")
            return {
                'average_marks': 0,
                'grade_distribution': {},
                'subject_averages': {},
                'total_students': 0
            }

    def export_data(self, format='excel', filename=None):
        """Export master data in specified format"""
        try:
            df = self.load_master_data()
            
            if df.empty:
                return {
                    'success': False,
                    'message': 'No data to export.'
                }
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'student_records_export_{timestamp}'
            
            if format.lower() == 'excel':
                filepath = f'data/{filename}.xlsx'
                df.to_excel(filepath, index=False)
            elif format.lower() == 'csv':
                filepath = f'data/{filename}.csv'
                df.to_csv(filepath, index=False)
            else:
                return {
                    'success': False,
                    'message': 'Unsupported export format. Use "excel" or "csv".'
                }
            
            return {
                'success': True,
                'message': f'Data exported successfully to {filepath}',
                'filepath': filepath,
                'record_count': len(df)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error exporting data: {str(e)}'
            }
    
    def search_student_by_id(self, student_id):
        """Search for a student by their ID"""
        try:
            df = self.load_master_data()
            
            if df.empty:
                return None
            
            # Search for exact match
            student = df[df['student_id'].astype(str) == str(student_id)]
            
            if not student.empty:
                return student.iloc[0].to_dict()
            
            return None
            
        except Exception as e:
            print(f"Error searching for student: {e}")
            return None
    
    def get_recent_data(self, limit=50):
        """Get recent data for display"""
        try:
            df = self.load_master_data()
            
            if df.empty:
                return []
            
            # Sort by timestamp if available, otherwise return last records
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp', ascending=False)
                except:
                    pass
            
            # Return recent records
            recent_data = df.head(limit)
            return recent_data.to_dict('records')
            
        except Exception as e:
            print(f"Error getting recent data: {e}")
            return [] 