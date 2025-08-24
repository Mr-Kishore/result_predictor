// Student Result Predictor - JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeForms();
    initializeFileUpload();
    initializeChatbot();
    initializeAnimations();
    initializeTooltips();
});

// Form handling and validation
function initializeForms() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        // Real-time validation
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', validateField);
            input.addEventListener('input', clearFieldError);
        });
        
        // Form submission
        form.addEventListener('submit', handleFormSubmit);
    });
}

function validateField(event) {
    const field = event.target;
    const value = field.value.trim();
    const fieldName = field.name;
    
    // Clear previous error
    clearFieldError(event);
    
    // Validation rules
    const rules = {
        'attendance_percentage': {
            required: true,
            min: 0,
            max: 100,
            message: 'Attendance must be between 0 and 100'
        },
        'assignment_marks': {
            required: true,
            min: 0,
            max: 100,
            message: 'Assignment marks must be between 0 and 100'
        },
        'midterm_marks': {
            required: true,
            min: 0,
            max: 100,
            message: 'Midterm marks must be between 0 and 100'
        },
        'final_exam_marks': {
            required: true,
            min: 0,
            max: 100,
            message: 'Final exam marks must be between 0 and 100'
        },
        'study_hours_per_day': {
            required: true,
            min: 0,
            max: 24,
            message: 'Study hours must be between 0 and 24'
        },
        'previous_semester_gpa': {
            required: true,
            min: 0,
            max: 4,
            message: 'GPA must be between 0 and 4'
        },
        'age': {
            required: false,
            min: 16,
            max: 30,
            message: 'Age must be between 16 and 30'
        },
        'family_income': {
            required: false,
            min: 0,
            message: 'Family income must be positive'
        }
    };
    
    const rule = rules[fieldName];
    if (!rule) return;
    
    if (rule.required && !value) {
        showFieldError(field, 'This field is required');
        return;
    }
    
    if (value && !isNaN(value)) {
        const numValue = parseFloat(value);
        if (rule.min !== undefined && numValue < rule.min) {
            showFieldError(field, rule.message);
            return;
        }
        if (rule.max !== undefined && numValue > rule.max) {
            showFieldError(field, rule.message);
            return;
        }
    }
}

function showFieldError(field, message) {
    field.classList.add('is-invalid');
    
    // Remove existing error message
    const existingError = field.parentNode.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
    
    // Add new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    field.parentNode.appendChild(errorDiv);
}

function clearFieldError(event) {
    const field = event.target;
    field.classList.remove('is-invalid');
    
    const errorDiv = field.parentNode.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

function handleFormSubmit(event) {
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    
    // Show loading state
    if (submitBtn) {
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        submitBtn.disabled = true;
        
        // Restore button after form submission
        setTimeout(() => {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }, 3000);
    }
}

// File upload functionality
function initializeFileUpload() {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.querySelector('input[type="file"]');
    
    if (uploadArea && fileInput) {
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        
        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // File selection
        fileInput.addEventListener('change', handleFileSelect);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFiles(files);
    }
}

function handleFiles(files) {
    const file = files[0];
    const allowedTypes = ['.xlsx', '.xls'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
        showAlert('Please select a valid Excel file (.xlsx or .xls)', 'danger');
        return;
    }
    
    // Show file info
    showFileInfo(file);
    
    // Auto-submit if form exists
    const form = document.querySelector('form');
    if (form) {
        form.submit();
    }
}

function showFileInfo(file) {
    const uploadArea = document.querySelector('.upload-area');
    if (uploadArea) {
        uploadArea.innerHTML = `
            <i class="fas fa-file-excel fa-3x text-success mb-3"></i>
            <h5>File Selected</h5>
            <p class="text-muted">${file.name}</p>
            <p class="text-muted">Size: ${formatFileSize(file.size)}</p>
            <div class="spinner mx-auto"></div>
            <p class="mt-2">Processing...</p>
        `;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Chatbot functionality
function initializeChatbot() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    
    if (chatForm && chatInput && chatMessages) {
        chatForm.addEventListener('submit', handleChatSubmit);
        
        // Auto-resize textarea
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });
        
        // Send message on Enter (but allow Shift+Enter for new line)
        chatInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleChatSubmit(event);
            }
        });
    }
}

function handleChatSubmit(event) {
    event.preventDefault();
    
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const message = chatInput.value.trim();
    
    if (!message) return;
    
    // Add user message
    addChatMessage(message, 'user');
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Show typing indicator
    const typingIndicator = addTypingIndicator();
    
    // Send to server
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Remove typing indicator
        typingIndicator.remove();
        
        if (data.success) {
            addChatMessage(data.response, 'bot');
            
            // Add suggestions if available
            if (data.suggestions && data.suggestions.length > 0) {
                addSuggestions(data.suggestions);
            }
        } else {
            addChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    })
    .catch(error => {
        typingIndicator.remove();
        addChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
        console.error('Chat error:', error);
    });
}

function addChatMessage(message, sender) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender} fade-in`;
    messageDiv.textContent = message;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing-indicator';
    typingDiv.innerHTML = '<i class="fas fa-ellipsis-h"></i> Typing...';
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingDiv;
}

function addSuggestions(suggestions) {
    const chatMessages = document.getElementById('chat-messages');
    const suggestionsDiv = document.createElement('div');
    suggestionsDiv.className = 'suggestions mt-2';
    
    suggestions.forEach(suggestion => {
        const btn = document.createElement('button');
        btn.className = 'btn btn-sm btn-outline-primary me-2 mb-1';
        btn.textContent = suggestion;
        btn.onclick = () => {
            document.getElementById('chat-input').value = suggestion;
            handleChatSubmit(new Event('submit'));
        };
        suggestionsDiv.appendChild(btn);
    });
    
    chatMessages.appendChild(suggestionsDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Animations
function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe all cards and sections
    document.querySelectorAll('.card, .row > div').forEach(el => {
        observer.observe(el);
    });
}

// Tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Utility functions
function showAlert(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertContainer, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertContainer.parentNode) {
                alertContainer.remove();
            }
        }, 5000);
    }
}

function showLoading(element) {
    if (element) {
        element.innerHTML = '<div class="spinner mx-auto"></div>';
        element.disabled = true;
    }
}

function hideLoading(element, originalText) {
    if (element) {
        element.innerHTML = originalText;
        element.disabled = false;
    }
}

// API functions
async function makePrediction(studentData, modelName = 'best') {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                student_data: studentData,
                model: modelName
            })
        });
        
        return await response.json();
    } catch (error) {
        console.error('Prediction error:', error);
        return { success: false, message: 'Network error' };
    }
}

async function uploadFile(file) {
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    } catch (error) {
        console.error('Upload error:', error);
        return { success: false, message: 'Network error' };
    }
}

async function getStats() {
    try {
        const response = await fetch('/api/stats');
        return await response.json();
    } catch (error) {
        console.error('Stats error:', error);
        return { success: false, message: 'Network error' };
    }
}

// Chart functions (if Chart.js is available)
function createChart(canvasId, data, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            ...options
        }
    });
}

// Export functions
function exportData(format = 'excel') {
    fetch('/api/export', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ format: format })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(`Data exported successfully to ${data.filepath}`, 'success');
        } else {
            showAlert(data.message, 'danger');
        }
    })
    .catch(error => {
        console.error('Export error:', error);
        showAlert('Export failed', 'danger');
    });
}

// Model comparison functions
async function getFeatureImportance(modelName = 'best') {
    try {
        const response = await fetch(`/api/feature-importance?model=${modelName}`);
        return await response.json();
    } catch (error) {
        console.error('Feature importance error:', error);
        return { success: false, message: 'Network error' };
    }
}

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    // Only show alert for critical errors, not all errors
    if (event.error && event.error.message && !event.error.message.includes('refreshData') && !event.error.message.includes('viewDataPreview') && !event.error.message.includes('downloadSample')) {
        showAlert('An unexpected error occurred', 'danger');
    }
});

// Service Worker registration (for PWA features) - disabled to prevent 404 errors
// if ('serviceWorker' in navigator) {
//     window.addEventListener('load', function() {
//         navigator.serviceWorker.register('/sw.js')
//             .then(function(registration) {
//                 console.log('SW registered: ', registration);
//             })
//             .catch(function(registrationError) {
//                 console.log('SW registration failed: ', registrationError);
//             });
//     });
// } 