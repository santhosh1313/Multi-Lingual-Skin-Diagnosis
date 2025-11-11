from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
import sqlite3
import hashlib
import os
import json
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from timm import create_model
import base64
from io import BytesIO

# Note: Using Browser TTS only - no IndicTrans2 or TTS library needed
# All translation and TTS handled client-side

app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
CORS(app, supports_credentials=True, origins=['http://localhost:5000', 'http://127.0.0.1:5000'], allow_headers=['Content-Type'], methods=['GET', 'POST', 'OPTIONS'])

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'SwinTransformer_best.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Class labels
CLASS_LABELS = [
    "Actinic Keratoses / Intraepithelial Carcinoma (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis-like Lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)"
]

# Short labels for display
SHORT_LABELS = {
    "akiec": "Actinic Keratoses (AK)",
    "bcc": "Basal Cell Carcinoma (BCC)",
    "bkl": "Benign Keratosis-like Lesions (BKL)",
    "df": "Dermatofibroma (DF)",
    "mel": "Melanoma (MEL)",
    "nv": "Melanocytic Nevi (NV)",
    "vasc": "Vascular Lesions (VASC)"
}

# Confirmation tests data
CONFIRMATION_TESTS = {
    "akiec": {
        "tests": "Skin biopsy, Dermoscopic examination",
        "description": "Doctors may take a small skin sample (biopsy) or use a special magnifying tool to confirm if it's a pre-cancerous spot."
    },
    "bcc": {
        "tests": "Skin biopsy, Dermatoscopy",
        "description": "A small part of the spot is removed and tested in a lab to confirm skin cancer. A close-up skin check using a dermatoscope may also be done."
    },
    "bkl": {
        "tests": "Dermoscopic examination, Biopsy (if needed)",
        "description": "Usually checked under a magnifying tool to make sure it's harmless. A biopsy is done only if it looks unusual."
    },
    "df": {
        "tests": "Clinical examination, Punch biopsy (if needed)",
        "description": "Doctors can usually identify it by looking and feeling the spot. A small sample (biopsy) is taken only if it changes quickly."
    },
    "mel": {
        "tests": "Excisional biopsy, Dermatoscopy",
        "description": "A full removal and lab test (biopsy) are done to confirm skin cancer. The doctor also checks for warning signs like changes in color, shape, or size."
    },
    "nv": {
        "tests": "Clinical monitoring, Dermatoscopy",
        "description": "These are common moles. Regular check-ups or photos help track changes. A dermatoscope helps tell if it's normal or needs attention."
    },
    "vasc": {
        "tests": "Dermoscopy, Ultrasound/Doppler scan",
        "description": "These are blood vessel-related spots. Doctors may use a magnifying lens or ultrasound to see how deep or active the blood vessels are."
    }
}

# Precautions data
PRECAUTIONS = {
    "akiec": [
        "Avoid direct sunlight during peak hours (10 AM–4 PM).",
        "Use sunscreen with SPF 30 or higher daily.",
        "Wear hats, sunglasses, and protective clothing when outdoors."
    ],
    "bcc": [
        "Protect your skin from sunburn and tanning beds.",
        "Get regular skin check-ups, especially if you've had BCC before.",
        "Treat any new or changing skin spots early to prevent spread."
    ],
    "bkl": [
        "Avoid scratching or picking at the lesion to prevent irritation.",
        "Keep skin clean and moisturized.",
        "Visit a dermatologist if you notice rapid growth or color change."
    ],
    "df": [
        "Do not try to remove or squeeze the bump yourself.",
        "Protect the area from repeated friction or injury.",
        "Get medical advice if it becomes painful or starts growing."
    ],
    "mel": [
        "Regularly check your moles using the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolution).",
        "Avoid tanning beds and prolonged UV exposure.",
        "See a dermatologist immediately for any changing or bleeding mole."
    ],
    "nv": [
        "Keep an eye on size, shape, and color changes.",
        "Limit sun exposure to prevent new or darkened moles.",
        "Visit a dermatologist for full-body mole mapping if you have many moles."
    ],
    "vasc": [
        "Avoid scratching or pressing the lesion to prevent bleeding.",
        "Use gentle skincare and avoid harsh chemicals on the area.",
        "Seek medical help if the lesion grows, changes color, or causes pain."
    ]
}

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  username TEXT)''')
    conn.commit()
    conn.close()

# Load model
global_model = None
def load_model():
    global global_model
    if global_model is None:
        try:
            global_model = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=len(CLASS_LABELS))
            global_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            global_model.to(DEVICE)
            global_model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            global_model = None
    return global_model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, BytesIO):
        image = Image.open(image).convert('RGB')
    return transform(image).unsqueeze(0).to(DEVICE)

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('.', 'skin.html')

@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirm_password')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400
        
        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'}), 400
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check if user already exists
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        if c.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'Email already registered'}), 400
        
        # Create user
        hashed_password = hash_password(password)
        username = email.split('@')[0]  # Use email prefix as username
        c.execute('INSERT INTO users (email, password, username) VALUES (?, ?, ?)',
                 (email, hashed_password, username))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Account created successfully'}), 201
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        hashed_password = hash_password(password)
        
        c.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, hashed_password))
        user = c.fetchone()
        conn.close()
        
        if user:
            user_id = user[0]
            user_email = user[1]
            user_username = user[3] if user[3] else email.split('@')[0]
            
            # Set session
            session['user_id'] = user_id
            session['email'] = user_email
            session['username'] = user_username
            
            # Make sure session is saved
            session.permanent = True
            
            print(f"Login successful - User ID: {user_id}, Username: {user_username}, Email: {user_email}")
            print(f"Session keys after login: {list(session.keys())}")
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'username': user_username,
                'email': user_email
            }), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200

@app.route('/api/user', methods=['GET'])
def get_user():
    if 'user_id' in session:
        username = session.get('username')
        email = session.get('email')
        # If username is not in session, try to get it from database
        if not username:
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('SELECT username, email FROM users WHERE id = ?', (session.get('user_id'),))
                user = c.fetchone()
                conn.close()
                if user:
                    username = user[0] if user[0] else email.split('@')[0] if email else 'User'
                    session['username'] = username
            except Exception as e:
                print(f"Error fetching username: {e}")
                username = email.split('@')[0] if email else 'User'
        
        return jsonify({
            'success': True,
            'username': username or 'User',
            'email': email or ''
        }), 200
    return jsonify({'success': False, 'message': 'Not logged in'}), 401

@app.route('/api/predict', methods=['POST'])
def predict():
    # Debug: print session info
    print(f"Session keys: {list(session.keys())}")
    print(f"User ID in session: {session.get('user_id')}")
    
    if 'user_id' not in session:
        print("No user_id in session, returning 401")
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    try:
        # Get image from request
        if 'image' not in request.files:
            # Try to get base64 image from JSON
            data = request.get_json()
            if data and 'image' in data:
                # Base64 image
                image_data = data['image']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            else:
                return jsonify({'success': False, 'message': 'No image provided'}), 400
        else:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'message': 'No file selected'}), 400
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'message': 'Invalid file type'}), 400
            image = Image.open(file.stream).convert('RGB')
        
        # Load model if not loaded
        prediction_model = load_model()
        if prediction_model is None:
            return jsonify({'success': False, 'message': 'Model not available'}), 500
        
        # Preprocess and predict
        img_tensor = preprocess_image(image)
        
        with torch.no_grad():
            outputs = prediction_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_idx].item() * 100
        
        # Get probabilities for all classes
        probabilities = {}
        for i, label in enumerate(CLASS_LABELS):
            prob = probs[0][i].item() * 100
            # Extract short code from label
            short_code = label.split('(')[-1].split(')')[0].strip()
            probabilities[short_code] = {
                'full_label': label,
                'short_label': SHORT_LABELS.get(short_code, label),
                'probability': prob
            }
        
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1]['probability'], reverse=True)
        
        # Get top predictions (top 1, and top 2 if > 1/4 of top 1)
        top1 = sorted_probs[0]
        top_predictions = [top1]
        
        if len(sorted_probs) > 1:
            top2_prob = sorted_probs[1][1]['probability']
            top1_prob = top1[1]['probability']
            if top2_prob > (top1_prob / 4):
                top_predictions.append(sorted_probs[1])
        
        # Prepare response
        result = {
            'success': True,
            'predictions': {},
            'top_predictions': [],
            'all_probabilities': {}
        }
        
        for code, data in probabilities.items():
            result['all_probabilities'][code] = {
                'label': data['short_label'],
                'probability': round(data['probability'], 2)
            }
        
        for code, data in top_predictions:
            result['top_predictions'].append({
                'code': code,
                'label': data['short_label'],
                'probability': round(data['probability'], 2),
                'tests': CONFIRMATION_TESTS.get(code, {}).get('tests', ''),
                'test_description': CONFIRMATION_TESTS.get(code, {}).get('description', ''),
                'precautions': PRECAUTIONS.get(code, [])
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

# Note: Translation and TTS are handled client-side using Browser TTS only
# No server-side models or API endpoints needed

if __name__ == '__main__':
    init_db()
    load_model()
    app.run(debug=True, port=5000)

