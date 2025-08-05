# Student Result Predictor

An advanced machine learning system to predict student academic performance using multiple algorithms and comprehensive data analysis.

##  Recent Updates & Bug Fixes

### Fixed Issues:
1. **Homepage Student Count Card** - Now clickable and redirects to student search page
2. **Student Search Functionality** - New search page with detailed student profiles and charts
3. **Improved Chatbot** - Enhanced with AI API integration (Hugging Face)
4. **Data Management** - Fixed data display issues in the management page
5. **Prediction Form** - Removed unnecessary fields (family income, parent education) and improved validation
6. **Better Error Handling** - Enhanced form validation and user feedback

##  Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, SVM, Neural Networks
- **Student Search**: Find students by roll number with detailed profiles and performance charts
- **AI Chatbot**: Intelligent assistant with API integration for better responses
- **Excel Integration**: Upload Excel files, automatic duplicate detection, batch processing
- **Analytics Dashboard**: Performance insights, feature importance, model comparison
- **Responsive Design**: Modern UI with Bootstrap 5

##  Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd REsult
```
**Creating a venv**:
```bash
python -m venv <venv-name>
```

***activating the venv**:
```bash
<venv-name>\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train the models** (if not already done):
```bash
python train_model.py
```

4. **Run the application**:
```bash
python app.py
```

5. **Access the application**:
   - Open your browser and go to `http://localhost:5000`

##  Environment Setup

### 1. **Create Environment File**:
   - Copy `env_example.txt` to `.env`:
   ```bash
   cp env_example.txt .env
   ```

### 2. **Configure API Keys**:
   - Edit the `.env` file and add your API keys:
   ```bash
   # Hugging Face API Token (get from https://huggingface.co/settings/tokens)
   HUGGINGFACE_TOKEN=your_actual_token_here
   
   # Flask Configuration
   SECRET_KEY=your-secret-key-here
   DEBUG=True
   ```

##  AI Chatbot Setup

To enable the enhanced AI chatbot with better responses:

1. **Get a free Hugging Face API key**:
   - Go to [Hugging Face](https://huggingface.co/)
   - Create a free account
   - Go to Settings → Access Tokens
   - Create a new token

2. **Configure the API key**:
   - Add your token to the `.env` file:
   ```bash
   HUGGINGFACE_TOKEN=your_actual_token_here
   ```

3. **Alternative AI APIs**:
   - You can also use other free AI APIs like:
     - OpenAI API (requires credit card)
     - Cohere API (free tier available)
     - Local models with Ollama

##  Usage

### 1. **Homepage**
- View system statistics and model performance
- Click on "Total Students" card to search for specific students
- Access quick actions for predictions, uploads, and chatbot

### 2. **Student Search**
- Enter a student's roll number to view detailed information
- See performance charts and AI predictions
- Get insights about strengths and areas for improvement

### 3. **Single Prediction**
- Enter individual student data
- Get instant predictions with confidence scores
- View performance insights and recommendations

### 4. **Batch Upload**
- Upload Excel files with multiple student records
- Automatic duplicate detection and processing
- Batch predictions for new records

### 5. **AI Assistant**
- Ask questions about the prediction system
- Get help with data interpretation
- Receive study advice and performance tips

### 6. **Data Management**
- View all student records
- Export data in various formats
- Monitor system statistics

##  Project Structure

```
REsult/
├── app.py                 # Main Flask application
├── chatbot/
│   └── chatbot.py        # AI chatbot implementation
├── utils/
│   ├── excel_handler.py  # Excel file processing
│   └── predictor.py      # ML prediction logic
├── templates/            # HTML templates
├── static/              # CSS, JS, and static files
├── data/                # Student data storage
├── model/               # Trained ML models
└── uploads/             # File upload directory
```

##  Configuration

### Environment Variables
- `FLASK_SECRET_KEY`: Secret key for Flask sessions
- `UPLOAD_FOLDER`: Directory for file uploads
- `MAX_CONTENT_LENGTH`: Maximum file upload size

### Model Configuration
- Models are automatically trained and saved in the `model/` directory
- The system selects the best performing model based on cross-validation
- Feature importance and model comparison are available

##  Model Performance

The system uses multiple machine learning algorithms:
- **Logistic Regression**: Fast and interpretable
- **Random Forest**: Robust and handles non-linear relationships
- **XGBoost**: High performance gradient boosting
- **Support Vector Machine**: Good for complex decision boundaries
- **Neural Network**: Deep learning approach

##  Troubleshooting

### Common Issues:

1. **Models not loading**:
   - Run `python train_model.py` to train models
   - Check if model files exist in `model/` directory

2. **Chatbot not responding**:
   - Verify API key configuration
   - Check internet connection for API calls
   - Fallback responses will work without API

3. **File upload issues**:
   - Ensure file is in Excel format (.xlsx, .xls)
   - Check file size limits
   - Verify required columns are present

4. **Prediction errors**:
   - Fill all required fields
   - Ensure numeric values are within valid ranges
   - Check for special characters in input

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Flask framework for web development
- Scikit-learn for machine learning algorithms
- Bootstrap for responsive UI design
- Hugging Face for AI model APIs
- Chart.js for data visualization

---

**Note**: This system is designed for educational purposes. Always ensure data privacy and security when handling student information.
