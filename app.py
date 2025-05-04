from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import cv2
from model_wrapper import H5ModelWrapper
from datetime import datetime
from functools import wraps
from io import BytesIO
import secrets
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = secrets.token_hex(16)  # Generate a secure random secret key
app.config['SESSION_COOKIE_SECURE'] = False  # Allow non-HTTPS connections for development
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session lifetime in seconds (1 hour)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model_path = os.path.join('models', 'hmpv_model (1).h5')
model = None
try:
    print(f"Attempting to load model from: {model_path}")
    # Use our custom model wrapper
    if os.path.exists(model_path):
        print("Model file exists, initializing wrapper...")
        model = H5ModelWrapper(model_path)
        print(f"Model wrapper initialized successfully with {model_path}")
    else:
        print(f"Model file not found at {model_path}. Using dummy model for testing.")
except Exception as e:
    print(f"Error initializing model wrapper: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    print("Using dummy model for testing")
    model = None

# Dummy data for demonstration
USERS = {
    'test@example.com': {'password': 'test123', 'name': 'Test User'}
}

DOCTORS = [
    {
        'id': 1,
        'name': 'John Smith',
        'specialty': 'Pulmonologist',
        'avatar': 'doctor1.svg',
        'online': True
    },
    {
        'id': 2,
        'name': 'Sarah Johnson',
        'specialty': 'Infectious Disease Specialist',
        'avatar': 'doctor2.svg',
        'online': True
    },
    {
        'id': 3,
        'name': 'Michael Brown',
        'specialty': 'General Physician',
        'avatar': 'doctor3.svg',
        'online': False
    },
    {
        'id': 4,
        'name': 'Emily Davis',
        'specialty': 'Radiologist',
        'avatar': 'doctor1.svg',
        'online': True
    }
]

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def process_image(image_path):
    """
    Process the uploaded image to match the model's input requirements
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        Processed image ready for model prediction
    """
    print(f"Processing image: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        raise ValueError(f"Could not read image at {image_path}")
    
    print(f"Original image shape: {img.shape}")
    
    # Basic X-ray validation
    # X-rays typically have a specific grayscale distribution
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # X-rays typically have a wider range of intensities and different distribution
    if mean_intensity < 50 or mean_intensity > 200 or std_intensity < 20:
        raise ValueError("The uploaded image does not appear to be a valid X-ray image. Please upload a clear X-ray image.")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to match your model's input size
    # Based on the model summary, the input size should be 70x70
    target_size = (70, 70)
    img = cv2.resize(img, target_size)
    print(f"Resized image shape: {img.shape}")
    
    # Normalize pixel values
    img = img / 255.0
    print(f"Normalized image min: {img.min()}, max: {img.max()}")
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    print(f"Final image shape for model: {img.shape}")
    
    # Clear any cached data
    tf.keras.backend.clear_session()
    
    return img

@app.route('/')
def index():
    if 'user' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            try:
                processed_image = process_image(filepath)
                
                # Make prediction
                if model is None:
                    flash('Model not loaded. Please contact support.', 'error')
                    return redirect(request.url)
                
                try:
                    # Use our model wrapper for prediction
                    prediction = model.predict(processed_image)
                    
                    # Get the class with highest probability
                    class_idx = np.argmax(prediction[0])
                    confidence = float(prediction[0][class_idx])
                    
                    # Map class index to label (binary HMPV result)
                    class_labels = ['HMPV Negative', 'HMPV Positive']
                    result_label = class_labels[class_idx]
                    
                    # Store results in session for report generation
                    session['prediction'] = result_label
                    session['confidence'] = confidence * 100
                    session['image_filename'] = filename
                    
                    flash('File successfully uploaded and analyzed', 'success')
                    
                    # Clear any cached data
                    tf.keras.backend.clear_session()
                    
                    return redirect(url_for('analyze'))
                except ValueError as e:
                    flash(str(e), 'error')
                    return redirect(request.url)
            except ValueError as e:
                # Handle X-ray validation errors
                flash(str(e), 'error')
                return redirect(request.url)
            except Exception as e:
                print(f"Error processing image: {e}")
                flash('Error processing image', 'error')
                return redirect(request.url)
        else:
            flash('File type not allowed', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email in USERS and USERS[email]['password'] == password:
            session.permanent = True  # Make the session permanent
            session['user'] = email
            print(f"User {email} logged in successfully")  # Debug logging
            return redirect(url_for('analyze'))
        else:
            flash('Invalid email or password')
            print(f"Failed login attempt for {email}")  # Debug logging
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Clear any session data if you're using sessions
    session.clear()
    # Redirect to the login page or home page
    return redirect(url_for('index'))

@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            try:
                processed_image = process_image(filepath)
                
                # Make prediction
                if model is not None:
                    try:
                        print("Model is loaded, making prediction...")
                        # Use our model wrapper for prediction
                        prediction = model.predict(processed_image)
                        
                        # Get the class with highest probability
                        class_idx = np.argmax(prediction[0])
                        confidence = float(prediction[0][class_idx])
                        
                        # Map class index to label (adjust based on your model's classes)
                        # Update to match all 4 classes from your model
                        class_labels = ['Normal', 'COVID-19', 'Viral Pneumonia', 'HMPV']
                        result_label = class_labels[class_idx]
                        print(f"Prediction successful: {result_label} with confidence {confidence}")
                    except Exception as e:
                        print(f"Error making prediction: {e}")
                        print(f"Error type: {type(e).__name__}")
                        import traceback
                        print(f"Full traceback: {traceback.format_exc()}")
                        result_label = 'Error analyzing image'
                        confidence = 0.0
                else:
                    # Use dummy prediction if model is not loaded
                    print("Model is None, using dummy prediction")
                    result_label = 'HMPV Positive (Dummy)'
                    confidence = 0.95
                
                # Store results in session for report generation
                session['prediction'] = result_label
                session['confidence'] = confidence * 100
                session['image_filename'] = filename
                
                # Prepare results for the template
                results = {
                    'condition': result_label,
                    'confidence': round(confidence * 100, 2),
                    'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low',
                    'details': f"""
                        <p>The analysis indicates a {result_label.lower()} result with {round(confidence * 100, 2)}% confidence.</p>
                        <p>This analysis is based on the visual patterns detected in the uploaded image using our advanced AI model.</p>
                    """,
                    'recommendations': [
                        "Schedule a consultation with a specialist for further evaluation",
                        "Monitor symptoms and keep track of any changes",
                        "Follow up with your healthcare provider as recommended",
                        "Maintain proper hygiene and follow preventive measures"
                    ]
                }
                
                # Clear any cached data
                tf.keras.backend.clear_session()
                
                return render_template('analysis.html', 
                                    image_filename=filename,
                                    results=results)
                
            except Exception as e:
                print(f"Error processing image: {e}")
                return jsonify({'error': 'Error processing image'}), 500
    # If GET, show the last analysis result from session if available
    if 'prediction' in session and 'confidence' in session and 'image_filename' in session:
        result_label = session['prediction']
        confidence = session['confidence'] / 100.0
        filename = session['image_filename']
        results = {
            'condition': result_label,
            'confidence': round(confidence * 100, 2),
            'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low',
            'details': f"""
                <p>The analysis indicates a {result_label.lower()} result with {round(confidence * 100, 2)}% confidence.</p>
                <p>This analysis is based on the visual patterns detected in the uploaded image using our advanced AI model.</p>
            """,
            'recommendations': [
                "Schedule a consultation with a specialist for further evaluation",
                "Monitor symptoms and keep track of any changes",
                "Follow up with your healthcare provider as recommended",
                "Maintain proper hygiene and follow preventive measures"
            ]
        }
        return render_template('analysis.html', 
                              image_filename=filename,
                              results=results)
    return render_template('index.html')

@app.route('/consultation')
@login_required
def consultation():
    # Get the latest analysis results from session
    prediction = session.get('prediction', 'Not Available')
    confidence = session.get('confidence', 0)
    
    # Map prediction to appropriate doctor recommendations
    if prediction == 'HMPV Negative':
        recommended_doctors = [d for d in DOCTORS if d['specialty'] == 'General Physician']
    elif prediction == 'HMPV Positive':
        recommended_doctors = [d for d in DOCTORS if d['specialty'] == 'Infectious Disease Specialist']
    else:
        recommended_doctors = DOCTORS
    
    return render_template('consultation.html', 
                         doctors=DOCTORS,
                         recommended_doctors=recommended_doctors,
                         prediction=prediction,
                         confidence=confidence)

@app.route('/generate-report')
@login_required
def generate_report():
    # Create a simple HTML report instead of PDF
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #007bff; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .recommendations {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <h1>Medical Analysis Report</h1>
        
        <h2>Patient Information</h2>
        <table>
            <tr>
                <th>Name</th>
                <td>{session.get('user', 'Not Available')}</td>
            </tr>
            <tr>
                <th>Date</th>
                <td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
            </tr>
            <tr>
                <th>Analysis Type</th>
                <td>Respiratory Condition Detection</td>
            </tr>
        </table>
        
        <h2>Analysis Results</h2>
        <table>
            <tr>
                <th>Prediction</th>
                <td>{session.get('prediction', 'Not Available')}</td>
            </tr>
            <tr>
                <th>Confidence</th>
                <td>{session.get('confidence', 0)}%</td>
            </tr>
        </table>
        
        <h2>Recommended Specialists</h2>
        <table>
            <tr>
                <th>Name</th>
                <th>Specialty</th>
                <th>Status</th>
            </tr>
    """
    
    # Get recommended doctors based on prediction
    prediction = session.get('prediction', 'Not Available')
    if prediction == 'Normal':
        recommended_doctors = [d for d in DOCTORS if d['specialty'] == 'General Physician']
    elif prediction == 'COVID-19' or prediction == 'Viral Pneumonia':
        recommended_doctors = [d for d in DOCTORS if d['specialty'] == 'Pulmonologist']
    elif prediction == 'HMPV':
        recommended_doctors = [d for d in DOCTORS if d['specialty'] == 'Infectious Disease Specialist']
    else:
        recommended_doctors = DOCTORS
    
    for doctor in recommended_doctors:
        report_html += f"""
            <tr>
                <td>{doctor['name']}</td>
                <td>{doctor['specialty']}</td>
                <td>{'Online' if doctor['online'] else 'Offline'}</td>
            </tr>
        """
    
    report_html += """
        </table>
        
        <h2>Recommendations</h2>
        <div class="recommendations">
            <p>1. Schedule a consultation with one of the recommended specialists.</p>
            <p>2. Keep track of your symptoms and any changes.</p>
            <p>3. Follow the specialist's advice regarding treatment and follow-up.</p>
            <p>4. Maintain proper hygiene and follow preventive measures.</p>
        </div>
    </body>
    </html>
    """
    
    # Create a BytesIO buffer for the HTML
    buffer = BytesIO()
    buffer.write(report_html.encode('utf-8'))
    buffer.seek(0)
    
    # Return the HTML as a downloadable file
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mimetype='text/html'
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        
        if email in USERS:
            flash('Email already registered')
        else:
            USERS[email] = {
                'password': password,
                'name': name
            }
            session['user'] = email
            return redirect(url_for('analyze'))
    
    return render_template('register.html')

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 