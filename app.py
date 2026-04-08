from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash, Response, send_from_directory, abort
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid
from config import Config
from backend.utils.auth import (
    get_user_by_username, get_user_by_email, create_user, verify_user, 
    create_or_get_firebase_user, verify_firebase_token, get_user_by_id,
    get_user_from_users, create_user_in_users, verify_user_in_users, get_user_by_id_from_users,
    check_username_exists, check_email_exists
)
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import smtplib
from email.message import EmailMessage
from backend.utils.db import init_db, get_db_connection, dict_from_row
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2
import time
import threading
import os
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash, Response, abort, send_from_directory
from backend.sign_detection.predict import predict_from_video_file, predict_from_bgr_frame, predict_from_landmarks
from backend.sign_detection.predict_image import predict_from_image_bytes as predict_from_image_bytes_cnn
from backend.sign_detection.model_loader import load_model
from collections import deque, defaultdict
import json
import datetime
import requests
from inference_sdk import InferenceHTTPClient
import os
from backend.sign_detection.model_loader import load_model as _load_model
from backend.sign_detection.language_model import build_lm_from_file
from backend.sign_detection.ctc_decode import beam_search_with_lm

# Global camera instance and lock for thread-safe access
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
        return camera

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.context_processor
def inject_user():
    user = None
    if 'user_id' in session:
        # Try users table first, then fallback to [user] table
        user = get_user_by_id_from_users(app.config['DATABASE'], session['user_id'])
        if not user:
            user = get_user_by_id(app.config['DATABASE'], session['user_id'])
    return dict(current_user=user)

def get_logged_in_user():
    if 'user_id' not in session:
        return None
    user = get_user_by_id_from_users(app.config['DATABASE'], session['user_id'])
    if user:
        user['_table'] = 'users'
        return user
    user = get_user_by_id(app.config['DATABASE'], session['user_id'])
    if user:
        user['_table'] = 'user'
        return user
    return None

# Session security
app.config.update(
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Ensure avatar upload dir exists
AVATAR_DIR = os.path.join(app.config['UPLOAD_FOLDER'], 'avatars')
os.makedirs(AVATAR_DIR, exist_ok=True)

ALLOWED_AVATAR_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'svg'}

def allowed_avatar(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AVATAR_EXTENSIONS

# serializer for email confirmation tokens
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Initialize database
init_db(app.config['DATABASE'])
# Database re-initialized

# Global model instances
MODEL = None
MODEL_YOLO = None
MODEL_MNIST = None

# Initialize Roboflow YOLO Inference Client
ROBOFLOW_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6zak5lO0xMEjWZSts5w6"
)
ROBOFLOW_WORKSPACE = "gajalakshmi-k"
ROBOFLOW_WORKFLOW_ID = "general-segmentation-api"

# Load model once at startup for real-time predictions (if available)
_model_path = app.config.get('MODEL_PATH')
_model_path_yolo = app.config.get('MODEL_PATH_YOLO')
_model_path_mnist = app.config.get('MODEL_PATH_MNIST')

try:
    MODEL = load_model(_model_path)
    if MODEL and _model_path:
        _labels_p = os.path.join(os.path.dirname(_model_path), 'labels.txt')
        if os.path.exists(_labels_p):
            with open(_labels_p, 'r') as f:
                MODEL.class_names = [l.strip() for l in f.readlines() if l.strip()]  # type: ignore
except Exception as e:
    print(f"Warning: Failed to load main model: {e}")
    MODEL = None

try:
    MODEL_YOLO = load_model(_model_path_yolo)
    if MODEL_YOLO and _model_path_yolo:
        print(f"YOLOv11 model loaded from {_model_path_yolo}")
except Exception as e:
    print(f"Warning: Failed to load YOLO model: {e}")
    MODEL_YOLO = None

try:
    MODEL_MNIST = load_model(_model_path_mnist)
    if MODEL_MNIST:
        print(f"Sign MNIST model loaded from {_model_path_mnist}")
except Exception as e:
    print(f"Warning: Failed to load MNIST model: {e}")
    MODEL_MNIST = None

# Preload per-language models if configured (MODEL_PATH_ASL, MODEL_PATH_BSL, ...)
LANGUAGES = ['asl', 'bsl', 'isl', 'lsf', 'ksl', 'jsl']
MODEL_LANG = {}
IMAGE_MODELS = {}
for _lang in LANGUAGES:
    try:
        p = app.config.get(f'MODEL_PATH_{_lang.upper()}')
        m = load_model(p) if p else None
        if m and p:
            _lp = os.path.join(os.path.dirname(p), 'labels.txt')
            if os.path.exists(_lp):
                with open(_lp, 'r') as f:
                    m.class_names = [l.strip() for l in f.readlines() if l.strip()]  # type: ignore 
        MODEL_LANG[_lang] = m
    except Exception:
        MODEL_LANG[_lang] = None
    try:
        ip = app.config.get(f'IMAGE_MODEL_PATH_{_lang.upper()}')
        if ip:
            try:
                from backend.sign_detection.predict_image import load_image_model
                m, cn = load_image_model(ip)
                IMAGE_MODELS[_lang] = {'model': m, 'class_names': cn}
            except Exception:
                IMAGE_MODELS[_lang] = {'model': load_model(ip), 'class_names': None}
        else:
            IMAGE_MODELS[_lang] = {'model': None, 'class_names': None}
    except Exception:
        IMAGE_MODELS[_lang] = {'model': None, 'class_names': None}

# Smoothing parameters (tuneable)
SMOOTH_WINDOW = int(os.environ.get('SMOOTH_WINDOW', 7))  # number of recent labels to keep
MAJORITY_THRESHOLD = int(os.environ.get('MAJORITY_THRESHOLD', 3))  # min count to accept majority

# Small in-memory smoothing store: map client key -> deque of recent labels
DETECTION_HISTORY = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))

# Landmark debug logging (set env LOG_LANDMARKS=1 to enable)
LOG_LANDMARKS = os.environ.get('LOG_LANDMARKS', '0') in ('1', 'true', 'yes')
LANDMARK_LOG_DIR = os.path.join(app.root_path, 'instance', 'landmarks')
os.makedirs(LANDMARK_LOG_DIR, exist_ok=True)

# Routes

# Simple label -> translated subtitle mapping for supported languages.
# This is a minimal approach: for production you'd want proper localization files
# or language-specific models. Keys are lowercase label tokens returned by
# prediction functions.
TRANSLATIONS = {
    'asl': {},
    'bsl': {},
    'isl': {},
    'lsf': {},
    'ksl': {},
    'jsl': {},
}

# Organized sign language data
asl_data = {
    "words": {
        'hello': {'asl': 'Hello', 'bsl': 'Hello', 'isl': 'नमस्ते', 'lsf': 'Bonjour', 'ksl': '안녕하세요', 'jsl': 'こんにちは'},
        'thank you': {'asl': 'Thank you', 'bsl': 'Thank you', 'isl': 'धन्यवाद', 'lsf': 'Merci', 'ksl': '감사합니다', 'jsl': 'ありがとうございます'},
        'please': {'asl': 'Please', 'bsl': 'Please', 'isl': 'कृपया', 'lsf': "S'il vous plaît", 'ksl': '부탁합니다', 'jsl': 'お願いします'},
        'good': {'asl': 'Good', 'bsl': 'Good', 'isl': 'अच्छा', 'lsf': 'Bon', 'ksl': '좋아요', 'jsl': '良い'},
        'no': {'asl': 'No', 'bsl': 'No', 'isl': 'नहीं', 'lsf': 'Non', 'ksl': '아니요', 'jsl': 'いいえ'},
        'yes': {'asl': 'Yes', 'bsl': 'Yes', 'isl': 'हाँ', 'lsf': 'Oui', 'ksl': '네', 'jsl': 'はい'},
        'help': {'asl': 'Help', 'bsl': 'Help', 'isl': 'मदद', 'lsf': 'Aide', 'ksl': '도와주세요', 'jsl': '助けて'},
        'sorry': {'asl': 'Sorry', 'bsl': 'Sorry', 'isl': 'माफ़ करना', 'lsf': 'Désolé', 'ksl': '미안해요', 'jsl': 'ごめんなさい'},
        'goodbye': {'asl': 'Goodbye', 'bsl': 'Goodbye', 'isl': 'अलविदा', 'lsf': 'Au revoir', 'ksl': '안녕히 가세요', 'jsl': 'さようなら'},
    },
    "alphabets": {
        'a': {'asl': 'A', 'bsl': 'A', 'isl': 'A', 'lsf': 'A', 'ksl': 'A', 'jsl': 'A'},
        'b': {'asl': 'B', 'bsl': 'B', 'isl': 'B', 'lsf': 'B', 'ksl': 'B', 'jsl': 'B'},
        'c': {'asl': 'C', 'bsl': 'C', 'isl': 'C', 'lsf': 'C', 'ksl': 'C', 'jsl': 'C'},
        'd': {'asl': 'D', 'bsl': 'D', 'isl': 'D', 'lsf': 'D', 'ksl': 'D', 'jsl': 'D'},
        'e': {'asl': 'E', 'bsl': 'E', 'isl': 'E', 'lsf': 'E', 'ksl': 'E', 'jsl': 'E'},
        'f': {'asl': 'F', 'bsl': 'F', 'isl': 'F', 'lsf': 'F', 'ksl': 'F', 'jsl': 'F'},
        'g': {'asl': 'G', 'bsl': 'G', 'isl': 'G', 'lsf': 'G', 'ksl': 'G', 'jsl': 'G'},
        'h': {'asl': 'H', 'bsl': 'H', 'isl': 'H', 'lsf': 'H', 'ksl': 'H', 'jsl': 'H'},
        'i': {'asl': 'I', 'bsl': 'I', 'isl': 'I', 'lsf': 'I', 'ksl': 'I', 'jsl': 'I'},
        'j': {'asl': 'J', 'bsl': 'J', 'isl': 'J', 'lsf': 'J', 'ksl': 'J', 'jsl': 'J'},
        'k': {'asl': 'K', 'bsl': 'K', 'isl': 'K', 'lsf': 'K', 'ksl': 'K', 'jsl': 'K'},
        'l': {'asl': 'L', 'bsl': 'L', 'isl': 'L', 'lsf': 'L', 'ksl': 'L', 'jsl': 'L'},
        'm': {'asl': 'M', 'bsl': 'M', 'isl': 'M', 'lsf': 'M', 'ksl': 'M', 'jsl': 'M'},
        'n': {'asl': 'N', 'bsl': 'N', 'isl': 'N', 'lsf': 'N', 'ksl': 'N', 'jsl': 'N'},
        'o': {'asl': 'O', 'bsl': 'O', 'isl': 'O', 'lsf': 'O', 'ksl': 'O', 'jsl': 'O'},
        'p': {'asl': 'P', 'bsl': 'P', 'isl': 'P', 'lsf': 'P', 'ksl': 'P', 'jsl': 'P'},
        'q': {'asl': 'Q', 'bsl': 'Q', 'isl': 'Q', 'lsf': 'Q', 'ksl': 'Q', 'jsl': 'Q'},
        'r': {'asl': 'R', 'bsl': 'R', 'isl': 'R', 'lsf': 'R', 'ksl': 'R', 'jsl': 'R'},
        's': {'asl': 'S', 'bsl': 'S', 'isl': 'S', 'lsf': 'S', 'ksl': 'S', 'jsl': 'S'},
        't': {'asl': 'T', 'bsl': 'T', 'isl': 'T', 'lsf': 'T', 'ksl': 'T', 'jsl': 'T'},
        'u': {'asl': 'U', 'bsl': 'U', 'isl': 'U', 'lsf': 'U', 'ksl': 'U', 'jsl': 'U'},
        'v': {'asl': 'V', 'bsl': 'V', 'isl': 'V', 'lsf': 'V', 'ksl': 'V', 'jsl': 'V'},
        'w': {'asl': 'W', 'bsl': 'W', 'isl': 'W', 'lsf': 'W', 'ksl': 'W', 'jsl': 'W'},
        'x': {'asl': 'X', 'bsl': 'X', 'isl': 'X', 'lsf': 'X', 'ksl': 'X', 'jsl': 'X'},
        'y': {'asl': 'Y', 'bsl': 'Y', 'isl': 'Y', 'lsf': 'Y', 'ksl': 'Y', 'jsl': 'Y'},
        'z': {'asl': 'Z', 'bsl': 'Z', 'isl': 'Z', 'lsf': 'Z', 'ksl': 'Z', 'jsl': 'Z'},
    },
    "numbers": {
        '0': {'asl': 'Zero (0)', 'bsl': 'Zero (0)', 'isl': 'शून्य (0)', 'lsf': 'Zéro (0)', 'ksl': '영 (0)', 'jsl': 'ゼロ (0)'},
        '1': {'asl': 'One (1)', 'bsl': 'One (1)', 'isl': 'एक (1)', 'lsf': 'Un (1)', 'ksl': '하나 (1)', 'jsl': '一 (1)'},
        '2': {'asl': 'Two (2)', 'bsl': 'Two (2)', 'isl': 'दो (2)', 'lsf': 'Deux (2)', 'ksl': '둘 (2)', 'jsl': '二 (2)'},
        '3': {'asl': 'Three (3)', 'bsl': 'Three (3)', 'isl': 'तीन (3)', 'lsf': 'Trois (3)', 'ksl': '셋 (3)', 'jsl': '三 (3)'},
        '4': {'asl': 'Four (4)', 'bsl': 'Four (4)', 'isl': 'चार (4)', 'lsf': 'Quatre (4)', 'ksl': '넷 (4)', 'jsl': '四 (4)'},
        '5': {'asl': 'Five (5)', 'bsl': 'Five (5)', 'isl': 'पाँच (5)', 'lsf': 'Cinq (5)', 'ksl': '다섯 (5)', 'jsl': '五 (5)'},
        '6': {'asl': 'Six (6)', 'bsl': 'Six (6)', 'isl': 'छह (6)', 'lsf': 'Six (6)', 'ksl': '여섯 (6)', 'jsl': '六 (6)'},
        '7': {'asl': 'Seven (7)', 'bsl': 'Seven (7)', 'isl': 'सात (7)', 'lsf': 'Sept (7)', 'ksl': '일곱 (7)', 'jsl': '七 (7)'},
        '8': {'asl': 'Eight (8)', 'bsl': 'Eight (8)', 'isl': 'आठ (8)', 'lsf': 'Huit (8)', 'ksl': '여덟 (8)', 'jsl': '八 (8)'},
        '9': {'asl': 'Nine (9)', 'bsl': 'Nine (9)', 'isl': 'नौ (9)', 'lsf': 'Neuf (9)', 'ksl': '아홉 (9)', 'jsl': '九 (9)'},
    }
}

# Flatten asl_data into TRANSLATIONS for quick lookup
for category in asl_data.values():
    for k, v in category.items():
        for lang_k, text in v.items():
            TRANSLATIONS.setdefault(lang_k, {})[k] = text

def translate_label(label, lang='asl'):
    """Return a language-specific subtitle for a predicted label.
    If no translation exists, return capitalized label or the label's sentence.
    """
    if label is None or label == '':
        return ''
    
    # Strip trailing period for lookup if it's a single word
    lab = str(label).lower().strip()
    if lab.endswith('.') and ' ' not in lab:
        lab = lab[:-1].strip()
    
    lang = (lang or 'asl').lower()
    
    # Check translations
    translation = TRANSLATIONS.get(lang, {}).get(lab)
    if translation:
        return translation
        
    if lab == 'space':
        return ' '
    if lab == 'del':
        return 'BACKSPACE'

    # If the label appears to be a full sentence (has spaces and not in translations), return it unchanged
    if ' ' in lab or (label.endswith('.') and len(lab) > 2):
        return label
        
    # For single characters or special tokens, don't add punctuation if not needed
    if len(lab) == 1:
        return label.upper()
        
    return label.capitalize() if isinstance(label, str) else str(label)

@app.route('/export/account.sql')
def export_account_sql():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Try getting from 'users' table first
    user = get_user_by_id_from_users(app.config['DATABASE'], session['user_id'])
    table_name = "users"
    
    if not user:
        # Fallback to older 'user' table
        user = get_user_by_id(app.config['DATABASE'], session['user_id'])
        table_name = "user"
        
    if not user:
        return abort(404)
        
    sql_content = f"-- Suhwa User Data Export ({table_name} table)\n"
    
    if table_name == "users":
        sql_content += "CREATE TABLE IF NOT EXISTS users (\n"
        sql_content += "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        sql_content += "    name TEXT NOT NULL,\n"
        sql_content += "    email TEXT UNIQUE NOT NULL,\n"
        sql_content += "    username TEXT UNIQUE NOT NULL,\n"
        sql_content += "    password_hash TEXT NOT NULL,\n"
        sql_content += "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
        sql_content += ");\n\n"
        cols = ['name', 'email', 'username', 'password_hash', 'created_at']
    else:
        sql_content += "CREATE TABLE IF NOT EXISTS [user] (\n"
        sql_content += "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        sql_content += "    username TEXT UNIQUE NOT NULL,\n"
        sql_content += "    password TEXT NOT NULL,\n"
        sql_content += "    email TEXT UNIQUE NOT NULL,\n"
        sql_content += "    first_name TEXT,\n"
        sql_content += "    last_name TEXT,\n"
        sql_content += "    dob TEXT,\n"
        sql_content += "    gender TEXT,\n"
        sql_content += "    is_confirmed INTEGER DEFAULT 0,\n"
        sql_content += "    confirm_token TEXT,\n"
        sql_content += "    firebase_uid TEXT UNIQUE,\n"
        sql_content += "    avatar TEXT,\n"
        sql_content += "    preferred_language TEXT,\n"
        sql_content += "    detection_mode TEXT,\n"
        sql_content += "    camera TEXT,\n"
        sql_content += "    output_type TEXT,\n"
        sql_content += "    hand_mode TEXT,\n"
        sql_content += "    camera_sensitivity INTEGER DEFAULT 700,\n"
        sql_content += "    total_signs INTEGER DEFAULT 0,\n"
        sql_content += "    last_detection TEXT,\n"
        sql_content += "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n"
        sql_content += "    last_login TIMESTAMP\n"
        sql_content += ");\n\n"
        cols = ['username', 'password', 'email', 'first_name', 'last_name', 'dob', 'gender', 'is_confirmed', 'confirm_token', 'avatar', 'preferred_language', 'detection_mode', 'camera', 'output_type', 'hand_mode', 'camera_sensitivity', 'total_signs', 'last_detection', 'created_at', 'last_login']
    
    vals = []
    for c in cols:
        val = user.get(c)
        if val is None:
            vals.append("NULL")
        elif isinstance(val, (int, float)):
            vals.append(str(val))
        else:
            # simple escaping for single quotes
            escaped = str(val).replace("'", "''")
            vals.append(f"'{escaped}'")
    
    sql_content += f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(vals)});\n"
    
    return Response(
        sql_content,
        mimetype="text/sql",
        headers={"Content-disposition": f"attachment; filename={table_name}_data.sql"}
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/detect')
def detect():
    user = get_logged_in_user()
    return render_template('detect.html', user=user)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        # Ensure filename is a string before passing to functions that expect str
        raw_filename = file.filename or ''
        if file and allowed_file(raw_filename):
            filename = secure_filename(raw_filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            start_time = time.time()
            try:
                result = predict_from_video_file(file_path, app.config['MODEL_PATH'])
                duration = time.time() - start_time
                
                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # Calculate word frequency
                words = result.get('words', [])
                word_frequency = {word: words.count(word) for word in set(words)} if words else {}
                
                # For simplicity, return JSON; frontend expects JSON response
                return jsonify({
                    'success': True,
                    'message': f"Predicted: {result.get('label', 'Unknown')} with confidence {result.get('confidence', 0)*100:.2f}%",
                    'result': result,
                    'translation': result.get('sentence', ''),
                    'english_translation': result.get('english_translation', ''),
                    'confidence': result.get('confidence', 0) * 100,
                    'total_signs': result.get('predictions_count', 0),
                    'duration': round(duration, 2),
                    'word_frequency': word_frequency,
                    'timeline': result.get('timeline', []),
                    'frames_analyzed': result.get('frames_analyzed', 0)
                })
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'success': False, 'message': f'Error processing video: {str(e)}'})
        else:
            flash('Invalid file type', 'error')
            return redirect(request.url)
    return render_template('upload.html')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/profile')
def profile():
    user = get_logged_in_user()
    if user is None:
        session.pop('user_id', None)
        flash('User account not found. Please login again.', 'error')
        return redirect(url_for('login'))
    username = user.get('username') or user.get('name') or 'User'
    user_profile_pic = user.get('avatar')
    return render_template('profile.html', username=username, user_profile_pic=user_profile_pic, user=user)


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    user = get_logged_in_user()
    if user is None:
        flash('User account not found. Please login again.', 'error')
        return redirect(url_for('login'))
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        dob = request.form.get('dob')
        gender = request.form.get('gender')
        preferred_language = request.form.get('preferred_language')
        detection_mode = request.form.get('detection_mode')
        camera_sel = request.form.get('camera')
        output_type = request.form.get('output_type')
        hand_mode = request.form.get('hand_mode')
        camera_sensitivity = request.form.get('camera_sensitivity')
        # simple uniqueness check for email
        existing = get_user_by_email(app.config['DATABASE'], email)
        if existing and user and existing.get('id') != user.get('id'):
            flash('Email already in use by another account.', 'error')
            return redirect(url_for('profile'))
        # update DB
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            if user.get('_table') == 'users':
                cursor.execute('UPDATE users SET first_name = ?, last_name = ?, name = ?, email = ?, dob = ?, gender = ?, preferred_language = ?, detection_mode = ?, camera = ?, output_type = ?, hand_mode = ?, camera_sensitivity = ? WHERE id = ?',
                               (first_name, last_name, f"{first_name} {last_name}".strip(), email, dob, gender, preferred_language, detection_mode, camera_sel, output_type, hand_mode, camera_sensitivity or None, user['id']))
            else:
                cursor.execute('UPDATE [user] SET first_name = ?, last_name = ?, email = ?, dob = ?, gender = ?, preferred_language = ?, detection_mode = ?, camera = ?, output_type = ?, hand_mode = ?, camera_sensitivity = ? WHERE id = ?',
                               (first_name, last_name, email, dob, gender, preferred_language, detection_mode, camera_sel, output_type, hand_mode, camera_sensitivity or None, user['id']))
            conn.commit()
            conn.close()
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Accept') == 'application/json':
                return jsonify({'success': True, 'message': 'Profile updated.'})
            flash('Profile updated.', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Accept') == 'application/json':
                return jsonify({'success': False, 'message': 'Failed to update profile: ' + str(e)})
            flash('Failed to update profile: ' + str(e), 'error')
            return redirect(url_for('profile'))
    return redirect(url_for('profile'))


@app.route('/upload_profile_pic', methods=['POST'])
def upload_profile_pic():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if 'profile_pic' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('profile'))
    file = request.files['profile_pic']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('profile'))
    if file and allowed_avatar(file.filename):
        raw_filename = file.filename or ''
        filename = secure_filename(raw_filename)
        # prefix with user id to avoid collisions
        filename = f"user_{session['user_id']}_" + filename
        file_path = os.path.join(AVATAR_DIR, filename)
        file.save(file_path)
        # persist in DB (store relative path under uploads/avatars)
        # store path using forward slashes so URLs work consistently across platforms
        rel_path = 'avatars/' + filename
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            user = get_logged_in_user()
            if user and user.get('_table') == 'users':
                cursor.execute('UPDATE users SET avatar = ? WHERE id = ?', (rel_path, session['user_id']))
            else:
                cursor.execute('UPDATE [user] SET avatar = ? WHERE id = ?', (rel_path, session['user_id']))
            conn.commit()
            conn.close()
            flash('Profile picture uploaded.', 'success')
        except Exception as e:
            flash('Failed to save profile picture: ' + str(e), 'error')
        return redirect(url_for('profile'))
    else:
        flash('Invalid file type for avatar.', 'error')
        return redirect(url_for('profile'))


@app.route('/set_avatar', methods=['POST'])
def set_avatar():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    preset = request.form.get('preset')
    if not preset:
        return jsonify({'success': False, 'message': 'No preset provided'}), 400
    # Only allow safe filenames (simple whitelist pattern)
    import re
    # allow svg as well for placeholder presets
    if not re.match(r'^[a-zA-Z0-9_\-]+\.(png|jpg|jpeg|gif|svg)$', preset):
        return jsonify({'success': False, 'message': 'Invalid filename'}), 400
    # store using forward slash for web-safe path
    rel_path = 'avatars/' + preset
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        user = get_logged_in_user()
        if user and user.get('_table') == 'users':
            cursor.execute('UPDATE users SET avatar = ? WHERE id = ?', (rel_path, session['user_id']))
        else:
            cursor.execute('UPDATE [user] SET avatar = ? WHERE id = ?', (rel_path, session['user_id']))
        conn.commit()
        conn.close()
        avatar_url = url_for('static', filename='uploads/' + rel_path)
        return jsonify({'success': True, 'avatar_url': avatar_url})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/view_activity')
def view_activity():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Placeholder activity view
    return render_template('activity.html')


@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    # Remove landmark logs and optionally sequence files
    try:
        # Clear landmark logs
        import glob
        removed = 0
        for f in glob.glob(os.path.join(LANDMARK_LOG_DIR, '*')):
            try:
                os.remove(f)
                removed += 1
            except Exception:
                pass
        return jsonify({'success': True, 'message': f'Cleared {removed} log files'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/save-translation', methods=['POST'])
def api_save_translation():
    data = request.get_json() or {}
    text = data.get('text') or ''
    confidence = data.get('confidence')
    raw = data.get('raw')
    ts = datetime.datetime.utcnow().isoformat()
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        uid = session.get('user_id')
        cursor.execute('INSERT INTO detection_history (user_id, timestamp, text, confidence, raw_label) VALUES (?, ?, ?, ?, ?)',
                       (uid, ts, text, confidence, raw))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect-stream', methods=['GET', 'POST'])
def api_detect_stream():
    """Consolidated sign detection endpoint.
    POST: handles client-side landmarks (used by WebRTC/Video Call).
    GET: handles server-side camera polling (used by standard Detect page).
    """
    lang = request.args.get('lang') or request.args.get('language') or 'asl'
    
    # Smoothing key (per user/session)
    try:
        key = session.get('user_id') or request.remote_addr
    except Exception:
        key = request.remote_addr

    result = {'success': False, 'subtitle': '', 'confidence': 0.0, 'label': None}

    if request.method == 'POST':
        data = request.get_json() or {}
        client_landmarks = data.get('landmarks')
        lang = data.get('lang', lang)
        
        if client_landmarks and len(client_landmarks) > 0:
            # We take the first hand's landmarks for classification
            hand = client_landmarks[0]
            # Flatten to [x, y, z, x, y, z, ...]
            lm_list = []
            for pt in hand:
                lm_list.extend([pt.get('x', 0), pt.get('y', 0), pt.get('z', 0)])
            
            lm_array = np.array(lm_list, dtype=np.float32)
            try:
                # Use language-specific model if available, fallback to default MODEL
                # or YOLO if mode is requested
                if lang == 'yolo' or request.args.get('mode') == 'yolo':
                    model_to_use = MODEL_YOLO or MODEL
                else:
                    model_to_use = MODEL_LANG.get(lang) or MODEL
                
                result = predict_from_landmarks(lm_array, model=model_to_use)
            except Exception as e:
                print(f"DEBUG: POST detect-stream error: {e}")
                return jsonify({'success': False, 'error': str(e)})
    else:
        # GET fallback: capture frame from server-side camera
        cam = get_camera()
        if cam and cam.isOpened():
            ret, frame = cam.read()
            if ret:
                try:
                    from backend.sign_detection.predict import predict_auto_from_bgr_frame
                    # Use YOLO model if explicitly requested or if it's the only available model for certain modes
                    if lang == 'yolo' or request.args.get('mode') == 'yolo':
                        try:
                            # Use Roboflow Inference Workflow for YOLO detection
                            # Save frame to temporary file as Roboflow client requires a path or URL
                            temp_image_path = os.path.join(app.root_path, 'temp_yolo_frame.jpg')
                            cv2.imwrite(temp_image_path, frame)
                            
                            workflow_result = ROBOFLOW_CLIENT.run_workflow(
                                workspace_name=ROBOFLOW_WORKSPACE,
                                workflow_id=ROBOFLOW_WORKFLOW_ID,
                                images={
                                    "image": temp_image_path
                                },
                                parameters={
                                    "classes": "A, B, C, D, E, F, G, H, I, J"
                                },
                                use_cache=True
                            )
                            
                            # Clean up temp file
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)

                            # Parse results - assuming it returns predictions in a standard format
                            # Extract the top prediction based on confidence
                            # Note: The structure of workflow_result depends on the Roboflow workflow definition
                            # Here we attempt to find detection-like results
                            predictions = []
                            if isinstance(workflow_result, list) and len(workflow_result) > 0:
                                output = workflow_result[0].get('outputs', {})
                                # Heuristic to find predictions in the output
                                for key, value in output.items():
                                    if isinstance(value, list) and len(value) > 0 and 'class_name' in value[0]:
                                        predictions = value
                                        break
                            
                            if predictions:
                                # Sort by confidence
                                top_pred = max(predictions, key=lambda x: x.get('confidence', 0))
                                result = {
                                    'success': True,
                                    'label': top_pred.get('class_name'),
                                    'confidence': top_pred.get('confidence', 0.0),
                                    'hand_count': len(predictions)
                                }
                            else:
                                result = {'success': False, 'label': None, 'confidence': 0.0, 'note': 'No detections'}
                                
                        except Exception as e:
                            print(f"DEBUG: Roboflow YOLO error: {e}")
                            model_to_use = MODEL_YOLO or MODEL
                            result = predict_auto_from_bgr_frame(frame, model=model_to_use)
                    else:
                        model_to_use = MODEL_LANG.get(lang) or MODEL
                        result = predict_auto_from_bgr_frame(frame, model=model_to_use)
                except Exception as e:
                    print(f"DEBUG: GET detect-stream error: {e}")
                    # fallback to older function
                    result = predict_from_bgr_frame(frame, model=MODEL)
            else:
                return jsonify({'success': False, 'error': 'Failed to capture frame'})
        else:
            return jsonify({'success': False, 'error': 'Camera not available'})

    # --- Smoothing & Majority Logic ---
    label = result.get('label')
    if label:
        DETECTION_HISTORY[key].append(label)
    
    counts = {}
    for l in DETECTION_HISTORY[key]:
        if l: counts[l] = counts.get(l, 0) + 1
    
    majority_label = None
    if counts:
        candidate, candidate_count = max(counts.items(), key=lambda kv: kv[1])
        if candidate_count >= MAJORITY_THRESHOLD:
            majority_label = candidate
            
    final_label = majority_label if majority_label is not None else label
    # Use sentence if available, else label
    sentence = result.get('sentence') if final_label == label else (final_label.capitalize() + '.' if final_label else '')
    
    subtitle = translate_label(sentence or final_label, lang=lang) if final_label else ''
    
    return jsonify({
        'success': True, 
        'subtitle': subtitle, 
        'label': final_label,
        'confidence': result.get('confidence', 0.0),
        'hand_count': result.get('hand_count', 1 if request.method == 'POST' else 0)
    })


@app.route('/api/chatbot')
def api_chatbot():
    q = request.args.get('q', '').lower()
    
    intents = {
        "greeting": {
            "keywords": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hai", "helo", "yo", "greetings", "whats up", "sup"],
            "response": "Hello! 👋 How can I help you with Suhwa today?"
        },
        "camera_issue": {
            "keywords": ["camera", "webcam", "camera not working", "no camera", "camera permission", "allow camera", "camera blocked", "video not showing", "camera error", "cam fix", "camera black", "cant see video", "webcam access"],
            "response": "Please enable camera permissions in your browser and ensure no other app is using the camera."
        },
        "upload_help": {
            "keywords": ["upload", "upload video", "file upload", "video upload", "mp4", "avi", "mov", "mkv", "upload failed", "file size", "video format", "upload limit", "how to upload", "send video"],
            "response": "You can upload MP4, AVI, MOV, or MKV files up to 2 minutes for best performance."
        },
        "supported_languages": {
            "keywords": ["language", "languages", "supported languages", "asl", "isl", "bsl", "sign languages", "which languages", "language support", "lsf", "ksl", "jsl", "french sign", "korean sign", "japanese sign", "indian sign", "american sign", "british sign"],
            "response": "We support ASL, ISL, BSL, LSF, KSL, and JSL. Check the Supported Languages page for more details."
        },
        "accuracy_tips": {
            "keywords": ["accuracy", "improve accuracy", "not detecting", "wrong output", "bad result", "prediction error", "gesture not recognized", "not working well", "fail", "slow detection", "blurry", "lighting", "background"],
            "response": "Ensure good lighting, a plain background, and keep your hands clearly visible within the frame."
        },
        "register_account": {
            "keywords": ["register", "sign up", "create account", "new user", "new account", "join", "how to join", "membership", "signup"],
            "response": "Click the Register button at the top-right corner to create a new account."
        },
        "login_help": {
            "keywords": ["login", "log in", "sign in", "password", "forgot password", "cannot login", "access account", "authentication", "signin"],
            "response": "Use your registered email and password on the Login page."
        },
        "logout_help": {
            "keywords": ["logout", "log out", "sign out", "exit account", "leave", "disconnect"],
            "response": "Click the Logout button in your dashboard to sign out safely."
        },
        "account_issue": {
            "keywords": ["account problem", "profile issue", "update profile", "change password", "edit profile", "avatar", "profile picture", "user settings", "delete account"],
            "response": "You can manage your profile, avatar, and password from the User Profile page."
        },
        "contact_support": {
            "keywords": ["contact", "support", "help desk", "email", "customer care", "technical support", "feedback", "report bug", "message us"],
            "response": "Contact us at support@suhwa.ai or through the Contact page."
        },
        "learning_mode": {
            "keywords": ["learn", "learning", "practice", "tutorial", "training", "lessons", "how to sign", "study", "education"],
            "response": "You can practice ASL alphabets, numbers, and words in Learning Mode."
        },
        "alphabets_help": {
            "keywords": ["alphabet", "alphabets", "a to z", "letters", "fingerspelling", "abc", "spelling", "sign letters"],
            "response": "To sign alphabets, keep your hand steady in front of your shoulder. Most signs are one-handed (ASL). For example, 'A' is a closed fist with thumb on the side."
        },
        "numbers_help": {
            "keywords": ["number", "numbers", "counting", "digits", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            "response": "ASL numbers 1-5 usually have palm facing inward. 6-9 involve the thumb touching different fingers."
        },
        "practice_advice": {
            "keywords": ["how to practice", "improve", "better", "tips for learning", "fast", "master"],
            "response": "Try our 'Practice with Camera' mode. It gives real-time feedback. Consistency is key - 10 minutes a day is better than an hour once a week!"
        },
        "alphabets_old": {
            "keywords": ["alphabets_old"],
            "response": "Our system detects all ASL alphabets (A-Z). Try the Alphabet Learning section to practice."
        },
        "numbers_help": {
            "keywords": ["numbers", "digits", "counting", "1 to 10", "1 to 20", "zero", "0-9", "sign numbers"],
            "response": "Our system detects numbers 0-9. Check the Numbers section in Learning Mode."
        },
        "video_call": {
            "keywords": ["video call", "call", "meet", "video meeting", "live call", "conference", "chat", "live chat"],
            "response": "You can start a video call using a shared link, allowing real-time sign language translation."
        },
        "subtitles": {
            "keywords": ["subtitles", "caption", "text output", "speech to text", "live captions", "translation text", "sub", "captions"],
            "response": "Live subtitles are displayed during detections and calls for better accessibility."
        },
        "performance_issue": {
            "keywords": ["slow", "lag", "delay", "performance", "loading issue", "freezing", "stutter", "hang"],
            "response": "Close background apps, ensure a stable internet connection, and check your browser hardware acceleration settings."
        },
        "features": {
            "keywords": ["features", "what can you do", "functionality", "tools", "capabilities", "services"],
            "response": "Suhwa offers real-time sign detection, video file translation, an AI tutor for learning, and accessible video calls."
        },
        "thanks": {
            "keywords": ["thanks", "thank you", "thx", "thankyou", "appreciate", "helpful"],
            "response": "You're welcome! 😊 Happy to help."
        },
        "goodbye": {
            "keywords": ["bye", "goodbye", "see you", "exit", "quit", "later", "cya"],
            "response": "Goodbye! 👋 Have a great day."
        }
    }
    
    answer = "I'm not sure about that. Try asking about 'camera', 'upload', 'languages', or 'how to learn sign language'."
    
    for intent in intents.values():
        if any(kw in q for kw in intent["keywords"]):
            answer = intent["response"]
            break
            
    return jsonify({'answer': answer})


@app.route('/api/get-history', methods=['GET'])
def api_get_history():
    # return last N history entries for current user
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        uid = session.get('user_id')
        
        # Determine if we are using SQL Server or SQLite for syntax
        is_sql_server = "DRIVER=" in app.config['DATABASE']
        
        if is_sql_server:
            if uid:
                cursor.execute('SELECT TOP 200 id, timestamp, text, confidence, raw_label FROM detection_history WHERE user_id = ? ORDER BY id DESC', (uid,))
            else:
                cursor.execute('SELECT TOP 200 id, timestamp, text, confidence, raw_label FROM detection_history WHERE user_id IS NULL ORDER BY id DESC')
        else:
            if uid:
                cursor.execute('SELECT id, timestamp, text, confidence, raw_label FROM detection_history WHERE user_id = ? ORDER BY id DESC LIMIT 200', (uid,))
            else:
                cursor.execute('SELECT id, timestamp, text, confidence, raw_label FROM detection_history WHERE user_id IS NULL ORDER BY id DESC LIMIT 200')
        
        rows = cursor.fetchall()
        conn.close()
        items = [{'id': r[0], 'timestamp': r[1], 'text': r[2], 'confidence': r[3], 'raw': r[4]} for r in rows]
        return jsonify({'success': True, 'history': items})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/export/history.txt')
def export_history_txt():
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        uid = session.get('user_id')
        if uid:
            cursor.execute('SELECT timestamp, text, confidence FROM detection_history WHERE user_id = ? ORDER BY id ASC', (uid,))
        else:
            cursor.execute('SELECT timestamp, text, confidence FROM detection_history WHERE user_id IS NULL ORDER BY id ASC')
        rows = cursor.fetchall()
        conn.close()
        lines = ["Suhwa Detection History Export", "="*30, ""]
        for ts, text, conf in rows:
            lines.append(f"[{ts}] (Confidence: {conf:.2f}) - {text}")
        data = '\n'.join(lines)
        return Response(data, mimetype='text/plain', headers={'Content-Disposition': 'attachment; filename=suhwa_history.txt'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/download/guide')
def download_guide():
    return send_from_directory(app.root_path, 'Suhwa_user_guide.pdf', as_attachment=True)


@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    user = get_logged_in_user()
    if user is None:
        # If user record is missing, force re-login to avoid attribute access errors
        flash('User account not found. Please login again.', 'error')
        return redirect(url_for('login'))
    if request.method == 'POST':
        old = request.form.get('old_password')
        new = request.form.get('new_password')
        new2 = request.form.get('new_password2')
        if not old or not new or new != new2:
            flash('Please provide matching new passwords.', 'error')
            return render_template('change_password.html')
        # verify old password
        verified = verify_user(app.config['DATABASE'], user.get('username'), old)
        if not verified:
            flash('Old password incorrect.', 'error')
            return render_template('change_password.html')
        # update password
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            # store secure hash
            pwdhash = generate_password_hash(new)
            if user.get('_table') == 'users':
                cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (pwdhash, user['id']))
            else:
                cursor.execute('UPDATE [user] SET password = ? WHERE id = ?', (pwdhash, user['id']))
            conn.commit()
            conn.close()
            flash('Password changed successfully.', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            flash('Failed to change password: ' + str(e), 'error')
            return render_template('change_password.html')
    return render_template('change_password.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/help-center')
def help_center():
    return render_template('help_center.html')

@app.route('/tutorials')
def tutorials():
    return render_template('tutorials.html')

@app.route('/community')
def community():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('community.html')

@app.route('/room/<int:room_id>')
def chat_room(room_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('chat_room.html', room_id=room_id)

@app.route('/api/current-user')
def api_current_user():
    user = get_logged_in_user()
    if user is None:
        return jsonify({'error': 'Not authenticated'}), 401
    return jsonify({
        'id': user['id'],
        'username': user.get('username') or user.get('name'),
        'email': user['email']
    })

@app.route('/api/get-rooms')
def api_get_rooms():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        is_sql_server = "DRIVER=" in app.config['DATABASE']
        
        cursor.execute('''SELECT DISTINCT cr.id, cr.room_name, cr.description, cr.room_type 
                         FROM chat_rooms cr
                         LEFT JOIN room_members rm ON cr.id = rm.room_id
                         WHERE cr.room_type = 'public' 
                         OR rm.user_id = ?
                         ORDER BY cr.created_at DESC''', (session['user_id'],))
        
        rooms_data = cursor.fetchall()
        rooms = []
        
        for row in rooms_data:
            room = dict_from_row(row, cursor)
            if room is None:
                continue
            
            cursor.execute('SELECT COUNT(*) FROM room_members WHERE room_id = ?', (room['id'],))
            member_count_row = cursor.fetchone()
            member_count = member_count_row[0] if member_count_row else 0
            room['member_count'] = member_count
            
            if is_sql_server:
                cursor.execute('''SELECT TOP 1 message_text FROM messages 
                                WHERE room_id = ? 
                                ORDER BY timestamp DESC''', (room['id'],))
            else:
                cursor.execute('''SELECT message_text FROM messages 
                                WHERE room_id = ? 
                                ORDER BY timestamp DESC LIMIT 1''', (room['id'],))
            last_msg = cursor.fetchone()
            room['last_message'] = last_msg[0][:50] if last_msg and last_msg[0] else 'No messages yet'
            
            rooms.append(room)
        
        if not rooms:
            cursor.execute("INSERT INTO chat_rooms (room_name, description, room_type) VALUES (?, ?, ?)", 
                          ('General', 'Welcome to the general discussion room', 'public'))
            conn.commit()
            cursor.execute('SELECT id, room_name, description, room_type FROM chat_rooms WHERE room_name = ?',
                          ('General',))
            room = dict_from_row(cursor.fetchone(), cursor)
            if room is not None:
                room['member_count'] = 0
                room['last_message'] = 'No messages yet'
                rooms = [room]
        
        conn.close()
        return jsonify(rooms)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-messages')
def api_get_messages():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room_id = request.args.get('room_id', type=int)
    if not room_id:
        return jsonify({'error': 'Missing room_id'}), 400
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        is_sql_server = "DRIVER=" in app.config['DATABASE']
        messages = []
        
        try:
            if is_sql_server:
                query = '''SELECT m.id, m.room_id, m.user_id, m.message_text, m.is_sign_detected, m.timestamp, 
                                  COALESCE(u.username, u2.username, 'Unknown User'), 
                                  ISNULL(m.message_type, 'text'), ISNULL(m.audio_url, ''), ISNULL(m.video_url, '')
                           FROM messages m
                           LEFT JOIN [user] u ON m.user_id = u.id
                           LEFT JOIN users u2 ON m.user_id = u2.id
                           WHERE m.room_id = ?
                           ORDER BY m.timestamp ASC'''
            else:
                query = '''SELECT m.id, m.room_id, m.user_id, m.message_text, m.is_sign_detected, m.timestamp, 
                                  COALESCE(u.username, u2.username, 'Unknown User'), 
                                  COALESCE(m.message_type, 'text'), COALESCE(m.audio_url, ''), COALESCE(m.video_url, '')
                           FROM messages m
                           LEFT JOIN [user] u ON m.user_id = u.id
                           LEFT JOIN users u2 ON m.user_id = u2.id
                           WHERE m.room_id = ?
                           ORDER BY m.timestamp ASC'''
            
            cursor.execute(query, (room_id,))
            
            rows = cursor.fetchall()
            print(f"Retrieved {len(rows)} messages for room {room_id}")
            
            for row in rows:
                messages.append({
                    'id': row[0],
                    'room_id': row[1],
                    'user_id': row[2],
                    'message_text': row[3],
                    'is_sign_detected': row[4],
                    'timestamp': str(row[5]),
                    'username': row[6],
                    'message_type': row[7] if row[7] else 'text',
                    'audio_url': row[8] if row[8] else None,
                    'video_url': row[9] if row[9] else None
                })
        except Exception as inner_e:
            print(f"Error with new query format: {inner_e}, trying legacy format...")
            cursor.execute('''SELECT m.id, m.room_id, m.user_id, m.message_text, m.is_sign_detected, m.timestamp, 
                                     COALESCE(u.username, u2.username, 'Unknown User')
                             FROM messages m
                             LEFT JOIN [user] u ON m.user_id = u.id
                             LEFT JOIN users u2 ON m.user_id = u2.id
                             WHERE m.room_id = ?
                             ORDER BY m.timestamp ASC''', (room_id,))
            
            rows = cursor.fetchall()
            print(f"Retrieved {len(rows)} messages for room {room_id} (legacy format)")
            
            for row in rows:
                messages.append({
                    'id': row[0],
                    'room_id': row[1],
                    'user_id': row[2],
                    'message_text': row[3],
                    'is_sign_detected': row[4],
                    'timestamp': str(row[5]),
                    'username': row[6],
                    'message_type': 'text',
                    'audio_url': None,
                    'video_url': None
                })
        
        conn.close()
        print(f"Returning {len(messages)} messages")
        return jsonify(messages)
    except Exception as e:
        print(f"Error in api_get_messages: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/create-room', methods=['POST'])
def api_create_room():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.json
    room_name = data.get('room_name', '').strip()
    description = data.get('description', '').strip()
    room_type = data.get('room_type', 'public')
    member_ids = data.get('member_ids', [])
    
    if not room_name:
        return jsonify({'success': False, 'error': 'Room name is required'}), 400
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''INSERT INTO chat_rooms (room_name, description, room_type, created_by)
                         VALUES (?, ?, ?, ?)''',
                      (room_name, description, room_type, session['user_id']))
        conn.commit()
        
        room_id = cursor.lastrowid  # type: ignore
        added_members = []
        
        cursor.execute('INSERT INTO room_members (room_id, user_id) VALUES (?, ?)',
                      (room_id, session['user_id']))
        added_members.append(session['user_id'])
        
        for member_id in member_ids:
            if member_id != session['user_id']:
                try:
                    cursor.execute('INSERT INTO room_members (room_id, user_id) VALUES (?, ?)',
                                  (room_id, member_id))
                    added_members.append(member_id)
                except Exception as e:
                    print(f"Error adding member {member_id}: {e}")
        
        conn.commit()
        conn.close()
        
        socketio.emit('room_created', {
            'room_id': room_id,
            'room_name': room_name,
            'description': description,
            'room_type': room_type
        })
        
        return jsonify({'success': True, 'room_id': room_id, 'members_added': len(added_members)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-all-users')
def api_get_all_users():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, username FROM [user] WHERE id != ? ORDER BY username',
                      (session['user_id'],))
        
        users = [{'id': row[0], 'username': row[1]} for row in cursor.fetchall()]
        conn.close()
        
        return jsonify(users)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-room-members')
def api_get_room_members():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    room_id = request.args.get('room_id', type=int)
    if not room_id:
        return jsonify({'error': 'Missing room_id'}), 400
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''SELECT u.id, u.username FROM room_members rm
                         JOIN [user] u ON rm.user_id = u.id
                         WHERE rm.room_id = ?''', (room_id,))
        
        members = [{'id': row[0], 'username': row[1]} for row in cursor.fetchall()]
        conn.close()
        
        return jsonify(members)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-room-members', methods=['POST'])
def api_add_room_members():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    try:
        room_id = int(data.get('room_id', 0))
    except (ValueError, TypeError):
        room_id = None
    user_ids = data.get('user_ids', [])
    
    if not room_id or not user_ids:
        return jsonify({'error': 'Missing room_id or user_ids'}), 400
    
    try:
        user_ids = [int(uid) for uid in user_ids if uid]
        if not user_ids:
            return jsonify({'error': 'Invalid user IDs'}), 400
        
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        added_members = []
        for user_id in user_ids:
            try:
                cursor.execute('''INSERT INTO room_members (room_id, user_id) VALUES (?, ?)''',
                              (room_id, user_id))
                added_members.append(user_id)
            except Exception as e:
                print(f"Error adding member {user_id} to room {room_id}: {e}")
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'members_added': len(added_members)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-voice-message', methods=['POST'])
def api_send_voice_message():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        room_id = int(request.form.get('room_id', 0))
        user_id = int(request.form.get('user_id', 0))
        
        if not room_id or not user_id:
            return jsonify({'error': 'Missing room_id or user_id'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        VOICE_UPLOAD_DIR = os.path.join(app.config['UPLOAD_FOLDER'], 'voice_messages')
        os.makedirs(VOICE_UPLOAD_DIR, exist_ok=True)
        
        filename = secure_filename(f"voice_{room_id}_{user_id}_{int(time.time() * 1000)}.webm")
        filepath = os.path.join(VOICE_UPLOAD_DIR, filename)
        audio_file.save(filepath)
        
        audio_url = f"/static/uploads/voice_messages/{filename}"
        
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO messages (room_id, user_id, message_text, message_type, audio_url, timestamp)
                         VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                      (room_id, user_id, '', 'audio', audio_url))
        conn.commit()
        message_id = cursor.lastrowid  # type: ignore
        conn.close()
        
        return jsonify({'success': True, 'message_id': message_id, 'audio_url': audio_url})
    except Exception as e:
        print(f"Error uploading voice message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-video-message', methods=['POST'])
def api_send_video_message():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        room_id = int(request.form.get('room_id', 0))
        user_id = int(request.form.get('user_id', 0))
        
        if not room_id or not user_id:
            return jsonify({'error': 'Missing room_id or user_id'}), 400
        
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        VIDEO_UPLOAD_DIR = os.path.join(app.config['UPLOAD_FOLDER'], 'video_messages')
        os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
        
        filename = secure_filename(f"video_{room_id}_{user_id}_{int(time.time() * 1000)}.webm")
        filepath = os.path.join(VIDEO_UPLOAD_DIR, filename)
        video_file.save(filepath)
        
        video_url = f"/static/uploads/video_messages/{filename}"
        
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO messages (room_id, user_id, message_text, message_type, video_url, timestamp)
                         VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                      (room_id, user_id, '', 'video', video_url))
        conn.commit()
        message_id = cursor.lastrowid  # type: ignore
        conn.close()
        
        return jsonify({'success': True, 'message_id': message_id, 'video_url': video_url})
    except Exception as e:
        print(f"Error uploading video message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-conversations')
def api_get_conversations():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''SELECT DISTINCT u.id, u.username
                         FROM private_messages pm
                         JOIN [user] u ON (pm.sender_id = u.id OR pm.receiver_id = u.id)
                         WHERE pm.sender_id = ? OR pm.receiver_id = ?
                         ORDER BY pm.timestamp DESC''',
                      (session['user_id'], session['user_id']))
        
        conversations = []
        seen_ids = set()
        for row in cursor.fetchall():
            user_id = row[0]
            if user_id not in seen_ids and user_id != session['user_id']:
                conversations.append({'id': user_id, 'username': row[1]})
                seen_ids.add(user_id)
        
        conn.close()
        return jsonify(conversations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-private-messages')
def api_get_private_messages():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''SELECT pm.id, pm.sender_id, pm.receiver_id, pm.message_text, pm.is_read, pm.is_sign_detected, pm.timestamp, u.username
                         FROM private_messages pm
                         JOIN [user] u ON pm.sender_id = u.id
                         WHERE (pm.sender_id = ? AND pm.receiver_id = ?) OR (pm.sender_id = ? AND pm.receiver_id = ?)
                         ORDER BY pm.timestamp ASC''',
                      (session['user_id'], user_id, user_id, session['user_id']))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'id': row[0],
                'sender_id': row[1],
                'receiver_id': row[2],
                'message_text': row[3],
                'is_read': row[4],
                'is_sign_detected': row[5],
                'timestamp': row[6],
                'username': row[7],
                'user_id': row[1]
            })
        
        conn.close()
        return jsonify(messages)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect_user')
def handle_connect(data):
    user_id = data.get('user_id')
    username = data.get('username')
    room_id = data.get('room_id')
    
    if user_id:
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM online_users WHERE user_id = ?', (user_id,))
            cursor.execute('INSERT INTO online_users (user_id, current_room_id) VALUES (?, ?)',
                          (user_id, room_id))
            conn.commit()
            conn.close()
            
            socketio.emit('update_online_users', {})
        except Exception as e:
            print(f"Error connecting user: {e}")

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data.get('room_id')
    user_id = data.get('user_id')
    
    if room_id:
        join_room(f'room_{room_id}')
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute('UPDATE online_users SET current_room_id = ? WHERE user_id = ?',
                          (room_id, user_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error joining room: {e}")

@socketio.on('send_message')
def handle_send_message(data):
    room_id = data.get('room_id')
    user_id = data.get('user_id')
    username = data.get('username')
    message_text = data.get('message_text')
    is_sign_detected = data.get('is_sign_detected', 0)
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''INSERT INTO messages (room_id, user_id, message_text, message_type, is_sign_detected, timestamp)
                         VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                      (room_id, user_id, message_text, 'text', is_sign_detected))
        conn.commit()
        
        msg_id = cursor.lastrowid  # type: ignore
        timestamp = datetime.datetime.now().isoformat()
        
        conn.close()
        
        print(f"Message saved - ID: {msg_id}, Room: {room_id}, User: {username}, Text: {message_text}")
        
        emit('new_message', {
            'id': msg_id,
            'room_id': room_id,
            'user_id': user_id,
            'username': username,
            'message_text': message_text,
            'message_type': 'text',
            'is_sign_detected': is_sign_detected,
            'timestamp': timestamp,
            'audio_url': None,
            'video_url': None
        }, to=f'room_{room_id}')
    except Exception as e:
        print(f"Error sending message: {e}")
        import traceback
        traceback.print_exc()

@socketio.on('user_typing')
def handle_user_typing(data):
    room_id = data.get('room_id')
    username = data.get('username')
    
    emit('user_typing', {
        'room_id': room_id,
        'username': username
    }, to=f'room_{room_id}')

@socketio.on('send_private_message')
def handle_private_message(data):
    receiver_id = data.get('receiver_id')
    message_text = data.get('message_text')
    is_sign_detected = data.get('is_sign_detected', 0)
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''INSERT INTO private_messages (sender_id, receiver_id, message_text, is_sign_detected)
                         VALUES (?, ?, ?, ?)''',
                      (session.get('user_id'), receiver_id, message_text, is_sign_detected))
        conn.commit()
        
        user = get_logged_in_user()
        timestamp = datetime.datetime.now().isoformat()
        
        conn.close()
        
        username = (user.get('username') or user.get('name')) if user else 'Unknown User'
        
        emit('new_private_message', {
            'sender_id': session['user_id'],
            'receiver_id': receiver_id,
            'username': username,
            'message_text': message_text,
            'is_sign_detected': is_sign_detected,
            'timestamp': timestamp,
            'user_id': session['user_id']
        })
    except Exception as e:
        print(f"Error sending private message: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('DELETE FROM online_users WHERE user_id = ?', (session.get('user_id'),))
        conn.commit()
        conn.close()
        socketio.emit('update_online_users', {})
    except Exception as e:
        print(f"Error disconnecting user: {e}")

@app.route('/reference-gallery')
def reference_gallery():
    """Visual Reference Gallery for alphabets and numbers."""
    user = get_logged_in_user()
    
    # List of reference images with metadata
    references = [
        {'title': 'BSL Alphabet', 'img': 'reference/british-sign-language-alphabet.png', 'lang': 'BSL'},
        {'title': 'BSL B-Sign', 'img': 'reference/british-sign-language-b.png', 'lang': 'BSL'},
        {'title': 'JSL Alphabet', 'img': 'reference/jsl-alphabets.png', 'lang': 'JSL'},
        {'title': 'JSL Basic', 'img': 'reference/jsl.png', 'lang': 'JSL'},
        {'title': 'KSL Alphabet', 'img': 'reference/ksl_korean.png', 'lang': 'KSL'},
        {'title': 'KSL Numbers', 'img': 'reference/ksl_korean_numbers.png', 'lang': 'KSL'},
        {'title': 'LSF Alphabet', 'img': 'reference/alphabet-french-sign-language-ls.png', 'lang': 'LSF'},
    ]
    
    return render_template('reference_gallery.html', user=user, references=references)


# Extended LESSONS list with Alphabets, Numbers, and Basic Words
LESSONS = [
    # ALPHABETS (A-Z)
    {"id": "asl_a", "title": "Alphabet: A", "language": "asl", "category": "alphabets", "target": "a", "emoji": "✊👍", "steps": [{"type": "intro", "text": "The letter <strong>A</strong> is a closed fist with the thumb on the side."}, {"type": "practice", "text": "Mimic the <strong>A</strong> sign."}]},
    {"id": "asl_b", "title": "Alphabet: B", "language": "asl", "category": "alphabets", "target": "b", "emoji": "🖐", "steps": [{"type": "intro", "text": "The letter <strong>B</strong> is an open palm with fingers together and thumb folded."}, {"type": "practice", "text": "Mimic the <strong>B</strong> sign."}]},
    {"id": "asl_c", "title": "Alphabet: C", "language": "asl", "category": "alphabets", "target": "c", "emoji": "🤏", "steps": [{"type": "intro", "text": "Curve your hand to form a <strong>C</strong> shape."}, {"type": "practice", "text": "Mimic the <strong>C</strong> sign."}]},
    {"id": "asl_d", "title": "Alphabet: D", "language": "asl", "category": "alphabets", "target": "d", "emoji": "☝️👌", "steps": [{"type": "intro", "text": "Index finger up, others touching thumb for <strong>D</strong>."}, {"type": "practice", "text": "Mimic the <strong>D</strong> sign."}]},
    {"id": "asl_e", "title": "Alphabet: E", "language": "asl", "category": "alphabets", "target": "e", "emoji": "🤜", "steps": [{"type": "intro", "text": "Fingers bent down touching thumb for <strong>E</strong>."}, {"type": "practice", "text": "Mimic the <strong>E</strong> sign."}]},
    {"id": "asl_f", "title": "Alphabet: F", "language": "asl", "category": "alphabets", "target": "f", "emoji": "👌", "steps": [{"type": "intro", "text": "Thumb and index touch in a circle for <strong>F</strong>."}, {"type": "practice", "text": "Mimic the <strong>F</strong> sign."}]},
    {"id": "asl_g", "title": "Alphabet: G", "language": "asl", "category": "alphabets", "target": "g", "emoji": "🤏➡️", "steps": [{"type": "intro", "text": "Thumb and index pointing sideways for <strong>G</strong>."}, {"type": "practice", "text": "Mimic the <strong>G</strong> sign."}]},
    {"id": "asl_h", "title": "Alphabet: H", "language": "asl", "category": "alphabets", "target": "h", "emoji": "✌️➡️", "steps": [{"type": "intro", "text": "Index and middle finger sideways for <strong>H</strong>."}, {"type": "practice", "text": "Mimic the <strong>H</strong> sign."}]},
    {"id": "asl_i", "title": "Alphabet: I", "language": "asl", "category": "alphabets", "target": "i", "emoji": "🤙", "steps": [{"type": "intro", "text": "Pinky finger up for <strong>I</strong>."}, {"type": "practice", "text": "Mimic the <strong>I</strong> sign."}]},
    {"id": "asl_j", "title": "Alphabet: J", "language": "asl", "category": "alphabets", "target": "j", "emoji": "🤙↩️", "steps": [{"type": "intro", "text": "Pinky up, draw letter <strong>J</strong> in the air."}, {"type": "practice", "text": "Mimic the <strong>J</strong> sign."}]},
    {"id": "asl_k", "title": "Alphabet: K", "language": "asl", "category": "alphabets", "target": "k", "emoji": "✌️👍", "steps": [{"type": "intro", "text": "Index and middle up, thumb between them for <strong>K</strong>."}, {"type": "practice", "text": "Mimic the <strong>K</strong> sign."}]},
    {"id": "asl_l", "title": "Alphabet: L", "language": "asl", "category": "alphabets", "target": "l", "emoji": "👍☝️", "steps": [{"type": "intro", "text": "Index up, thumb sideways for <strong>L</strong> shape."}, {"type": "practice", "text": "Mimic the <strong>L</strong> sign."}]},
    {"id": "asl_m", "title": "Alphabet: M", "language": "asl", "category": "alphabets", "target": "m", "emoji": "✊", "steps": [{"type": "intro", "text": "Thumb under 3 fingers for <strong>M</strong>."}, {"type": "practice", "text": "Mimic the <strong>M</strong> sign."}]},
    {"id": "asl_n", "title": "Alphabet: N", "language": "asl", "category": "alphabets", "target": "n", "emoji": "✊", "steps": [{"type": "intro", "text": "Thumb under 2 fingers for <strong>N</strong>."}, {"type": "practice", "text": "Mimic the <strong>N</strong> sign."}]},
    {"id": "asl_o", "title": "Alphabet: O", "language": "asl", "category": "alphabets", "target": "o", "emoji": "🫶", "steps": [{"type": "intro", "text": "Fingers touch thumb making a circle for <strong>O</strong>."}, {"type": "practice", "text": "Mimic the <strong>O</strong> sign."}]},
    {"id": "asl_p", "title": "Alphabet: P", "language": "asl", "category": "alphabets", "target": "p", "emoji": "✌️⬇️", "steps": [{"type": "intro", "text": "Like K but facing downward for <strong>P</strong>."}, {"type": "practice", "text": "Mimic the <strong>P</strong> sign."}]},
    {"id": "asl_q", "title": "Alphabet: Q", "language": "asl", "category": "alphabets", "target": "q", "emoji": "🤏⬇️", "steps": [{"type": "intro", "text": "Like G but facing downward for <strong>Q</strong>."}, {"type": "practice", "text": "Mimic the <strong>Q</strong> sign."}]},
    {"id": "asl_r", "title": "Alphabet: R", "language": "asl", "category": "alphabets", "target": "r", "emoji": "🤞", "steps": [{"type": "intro", "text": "Cross index and middle fingers for <strong>R</strong>."}, {"type": "practice", "text": "Mimic the <strong>R</strong> sign."}]},
    {"id": "asl_s", "title": "Alphabet: S", "language": "asl", "category": "alphabets", "target": "s", "emoji": "✊", "steps": [{"type": "intro", "text": "Tight fist, thumb in front for <strong>S</strong>."}, {"type": "practice", "text": "Mimic the <strong>S</strong> sign."}]},
    {"id": "asl_t", "title": "Alphabet: T", "language": "asl", "category": "alphabets", "target": "t", "emoji": "🤏", "steps": [{"type": "intro", "text": "Thumb between index and middle for <strong>T</strong>."}, {"type": "practice", "text": "Mimic the <strong>T</strong> sign."}]},
    {"id": "asl_u", "title": "Alphabet: U", "language": "asl", "category": "alphabets", "target": "u", "emoji": "✌️", "steps": [{"type": "intro", "text": "Index and middle together up for <strong>U</strong>."}, {"type": "practice", "text": "Mimic the <strong>U</strong> sign."}]},
    {"id": "asl_v", "title": "Alphabet: V", "language": "asl", "category": "alphabets", "target": "v", "emoji": "✌️", "steps": [{"type": "intro", "text": "Index and middle spread apart for <strong>V</strong>."}, {"type": "practice", "text": "Mimic the <strong>V</strong> sign."}]},
    {"id": "asl_w", "title": "Alphabet: W", "language": "asl", "category": "alphabets", "target": "w", "emoji": "🖖", "steps": [{"type": "intro", "text": "Three fingers up for <strong>W</strong>."}, {"type": "practice", "text": "Mimic the <strong>W</strong> sign."}]},
    {"id": "asl_x", "title": "Alphabet: X", "language": "asl", "category": "alphabets", "target": "x", "emoji": "☝️", "steps": [{"type": "intro", "text": "Index bent like a hook for <strong>X</strong>."}, {"type": "practice", "text": "Mimic the <strong>X</strong> sign."}]},
    {"id": "asl_y", "title": "Alphabet: Y", "language": "asl", "category": "alphabets", "target": "y", "emoji": "🤙", "steps": [{"type": "intro", "text": "Thumb and pinky out for <strong>Y</strong>."}, {"type": "practice", "text": "Mimic the <strong>Y</strong> sign."}]},
    {"id": "asl_z", "title": "Alphabet: Z", "language": "asl", "category": "alphabets", "target": "z", "emoji": "☝️➡️⬅️", "steps": [{"type": "intro", "text": "Draw <strong>Z</strong> in the air with index finger."}, {"type": "practice", "text": "Mimic the <strong>Z</strong> sign."}]},

    # NUMBERS (0-10)
    {"id": "num_0", "title": "Number: 0", "language": "asl", "category": "numbers", "target": "0", "emoji": "🫶", "steps": [{"type": "intro", "text": "Form an <strong>O</strong> shape for zero."}, {"type": "practice", "text": "Show the number 0."}]},
    {"id": "num_1", "title": "Number: 1", "language": "asl", "category": "numbers", "target": "1", "emoji": "☝️", "steps": [{"type": "intro", "text": "Index finger up for one."}, {"type": "practice", "text": "Show the number 1."}]},
    {"id": "num_2", "title": "Number: 2", "language": "asl", "category": "numbers", "target": "2", "emoji": "✌️", "steps": [{"type": "intro", "text": "Index and middle up for two."}, {"type": "practice", "text": "Show the number 2."}]},
    {"id": "num_3", "title": "Number: 3", "language": "asl", "category": "numbers", "target": "3", "emoji": "🤟", "steps": [{"type": "intro", "text": "Thumb, index, and middle for three."}, {"type": "practice", "text": "Show the number 3."}]},
    {"id": "num_4", "title": "Number: 4", "language": "asl", "category": "numbers", "target": "4", "emoji": "🖐", "steps": [{"type": "intro", "text": "Four fingers up for four."}, {"type": "practice", "text": "Show the number 4."}]},
    {"id": "num_5", "title": "Number: 5", "language": "asl", "category": "numbers", "target": "5", "emoji": "✋", "steps": [{"type": "intro", "text": "Open palm for five."}, {"type": "practice", "text": "Show the number 5."}]},
    {"id": "num_6", "title": "Number: 6", "language": "asl", "category": "numbers", "target": "6", "emoji": "🤙", "steps": [{"type": "intro", "text": "Thumb touches pinky for six."}, {"type": "practice", "text": "Show the number 6."}]},
    {"id": "num_7", "title": "Number: 7", "language": "asl", "category": "numbers", "target": "7", "emoji": "🤏", "steps": [{"type": "intro", "text": "Thumb touches ring finger for seven."}, {"type": "practice", "text": "Show the number 7."}]},
    {"id": "num_8", "title": "Number: 8", "language": "asl", "category": "numbers", "target": "8", "emoji": "🤏", "steps": [{"type": "intro", "text": "Thumb touches middle finger for eight."}, {"type": "practice", "text": "Show the number 8."}]},
    {"id": "num_9", "title": "Number: 9", "language": "asl", "category": "numbers", "target": "9", "emoji": "👌", "steps": [{"type": "intro", "text": "Thumb touches index for nine."}, {"type": "practice", "text": "Show the number 9."}]},
    {"id": "num_10", "title": "Number: 10", "language": "asl", "category": "numbers", "target": "10", "emoji": "✊🔄", "steps": [{"type": "intro", "text": "Close fist and shake for ten."}, {"type": "practice", "text": "Show the number 10."}]},

    # BASIC WORDS (20)
    {"id": "word_hello", "title": "Word: Hello", "language": "asl", "category": "words", "target": "hello", "emoji": "👋", "steps": [{"type": "intro", "text": "Wave your hand for <strong>Hello</strong>."}, {"type": "practice", "text": "Say Hello!"}]},
    {"id": "word_thankyou", "title": "Word: Thank You", "language": "asl", "category": "words", "target": "thank you", "emoji": "🙏➡️", "steps": [{"type": "intro", "text": "Fingers from chin forward for <strong>Thank You</strong>."}, {"type": "practice", "text": "Say Thank You!"}]},
    {"id": "word_please", "title": "Word: Please", "language": "asl", "category": "words", "target": "please", "emoji": "🤲🔄", "steps": [{"type": "intro", "text": "Rub chest in a circular motion for <strong>Please</strong>."}, {"type": "practice", "text": "Say Please!"}]},
    {"id": "word_sorry", "title": "Word: Sorry", "language": "asl", "category": "words", "target": "sorry", "emoji": "✊🔄", "steps": [{"type": "intro", "text": "Fist circle on chest for <strong>Sorry</strong>."}, {"type": "practice", "text": "Say Sorry!"}]},
    {"id": "word_yes", "title": "Word: Yes", "language": "asl", "category": "words", "target": "yes", "emoji": "✊⬆️⬇️", "steps": [{"type": "intro", "text": "Fist nod up and down for <strong>Yes</strong>."}, {"type": "practice", "text": "Say Yes!"}]},
    {"id": "word_no", "title": "Word: No", "language": "asl", "category": "words", "target": "no", "emoji": "✌️🤏", "steps": [{"type": "intro", "text": "Index and middle finger tap thumb for <strong>No</strong>."}, {"type": "practice", "text": "Say No!"}]},
    {"id": "word_friend", "title": "Word: Friend", "language": "asl", "category": "words", "target": "friend", "emoji": "🤞🤝", "steps": [{"type": "intro", "text": "Hook index fingers together for <strong>Friend</strong>."}, {"type": "practice", "text": "Say Friend!"}]},
    {"id": "word_love", "title": "Word: Love", "language": "asl", "category": "words", "target": "love", "emoji": "🤗❤️", "steps": [{"type": "intro", "text": "Cross arms over chest for <strong>Love</strong>."}, {"type": "practice", "text": "Say Love!"}]},
    {"id": "word_home", "title": "Word: Home", "language": "asl", "category": "words", "target": "home", "emoji": "🤚🏠", "steps": [{"type": "intro", "text": "Touch cheek twice for <strong>Home</strong>."}, {"type": "practice", "text": "Say Home!"}]},
    {"id": "word_mother", "title": "Word: Mother", "language": "asl", "category": "words", "target": "mother", "emoji": "👍👩", "steps": [{"type": "intro", "text": "Thumb on chin for <strong>Mother</strong>."}, {"type": "practice", "text": "Say Mother!"}]},
    {"id": "word_father", "title": "Word: Father", "language": "asl", "category": "words", "target": "father", "emoji": "👍👨", "steps": [{"type": "intro", "text": "Thumb on forehead for <strong>Father</strong>."}, {"type": "practice", "text": "Say Father!"}]},
    {"id": "word_teacher", "title": "Word: Teacher", "language": "asl", "category": "words", "target": "teacher", "emoji": "🖐➡️", "steps": [{"type": "intro", "text": "Both hands open outward for <strong>Teacher</strong>."}, {"type": "practice", "text": "Say Teacher!"}]},
    {"id": "word_study", "title": "Word: Study", "language": "asl", "category": "words", "target": "study", "emoji": "📖🖐", "steps": [{"type": "intro", "text": "Palm brush over other palm for <strong>Study</strong>."}, {"type": "practice", "text": "Say Study!"}]},
    {"id": "word_eat", "title": "Word: Eat", "language": "asl", "category": "words", "target": "eat", "emoji": "🤏🍽", "steps": [{"type": "intro", "text": "Fingers to mouth for <strong>Eat</strong>."}, {"type": "practice", "text": "Say Eat!"}]},
    {"id": "word_water", "title": "Word: Water", "language": "asl", "category": "words", "target": "water", "emoji": "🖖💧", "steps": [{"type": "intro", "text": "W shape near mouth for <strong>Water</strong>."}, {"type": "practice", "text": "Say Water!"}]},
    {"id": "word_stop", "title": "Word: Stop", "language": "asl", "category": "words", "target": "stop", "emoji": "✋🛑", "steps": [{"type": "intro", "text": "One palm hits the other for <strong>Stop</strong>."}, {"type": "practice", "text": "Say Stop!"}]},
    {"id": "word_help", "title": "Word: Help", "language": "asl", "category": "words", "target": "help", "emoji": "✊🤲⬆️", "steps": [{"type": "intro", "text": "Fist on palm, lift upward for <strong>Help</strong>."}, {"type": "practice", "text": "Say Help!"}]},
    {"id": "word_run", "title": "Word: Run", "language": "asl", "category": "words", "target": "run", "emoji": "✌️🏃", "steps": [{"type": "intro", "text": "Two fingers running motion for <strong>Run</strong>."}, {"type": "practice", "text": "Say Run!"}]},
    {"id": "word_sleep", "title": "Word: Sleep", "language": "asl", "category": "words", "target": "sleep", "emoji": "😴🛏", "steps": [{"type": "intro", "text": "Hands together near cheek for <strong>Sleep</strong>."}, {"type": "practice", "text": "Say Sleep!"}]},
    {"id": "word_happy", "title": "Word: Happy", "language": "asl", "category": "words", "target": "happy", "emoji": "😊⬆️", "steps": [{"type": "intro", "text": "Brush hands upward on chest for <strong>Happy</strong>."}, {"type": "practice", "text": "Say Happy!"}]}
]

@app.route('/api/ai/lessons')
def api_get_lessons():
    """List all available lessons."""
    return jsonify({'success': True, 'lessons': LESSONS})

@app.route('/api/ai/lesson/<lesson_id>')
def api_get_lesson(lesson_id):
    """Get details for a specific lesson."""
    lesson = next((l for l in LESSONS if l['id'] == lesson_id), None)
    if not lesson:
        return jsonify({'success': False, 'message': 'Lesson not found'}), 404
    
    # Get user progress if logged in
    progress = None
    if 'user_id' in session:
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute('SELECT status, accuracy, attempts FROM user_progress WHERE user_id = ? AND lesson_id = ?', 
                           (session['user_id'], lesson_id))
            row = cursor.fetchone()
            if row:
                progress = {'status': row[0], 'accuracy': row[1], 'attempts': row[2]}
            conn.close()
        except Exception:
            pass
            
    return jsonify({'success': True, 'lesson': lesson, 'progress': progress})

@app.route('/api/ai/submit-practice', methods=['POST'])
def api_submit_practice():
    """Evaluate practice attempt and update progress."""
    data = request.get_json() or {}
    lesson_id = data.get('lesson_id')
    predicted_label = data.get('label', '').lower()
    confidence = data.get('confidence', 0.0)
    
    lesson = next((l for l in LESSONS if l['id'] == lesson_id), None)
    if not lesson:
        return jsonify({'success': False, 'message': 'Lesson not found'}), 404
    
    target = lesson['target'].lower()
    is_correct = predicted_label == target
    
    xp_gained = 0
    if is_correct:
        # 10 XP base, plus confidence bonus
        xp_gained = 10 + int(confidence * 10)

    feedback = ""
    if is_correct:
        feedback = f"Great job! You signed '{target}' correctly with {confidence*100:.1f}% confidence. +{xp_gained} XP!"
    else:
        feedback = f"Not quite. You signed '{predicted_label}', but we were looking for '{target}'. Try again!"

    # Update progress in DB if logged in
    if 'user_id' in session:
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            
            # Update user XP and Streak
            today = datetime.datetime.utcnow().date()
            cursor.execute('SELECT xp_points, daily_streak, last_practice_date FROM [user] WHERE id = ?', (session['user_id'],))
            u_row = cursor.fetchone()
            if u_row:
                old_xp, old_streak, last_date = u_row
                new_xp = old_xp + xp_gained
                new_streak = old_streak
                
                # Simple streak logic
                if last_date:
                    try:
                        # Assuming last_date might be string or datetime
                        if isinstance(last_date, str):
                            last_date_obj = datetime.datetime.fromisoformat(last_date).date()
                        else:
                            last_date_obj = last_date.date()
                        
                        if last_date_obj == today:
                            pass # Already practiced today
                        elif last_date_obj == today - datetime.timedelta(days=1):
                            new_streak += 1
                        else:
                            new_streak = 1 if xp_gained > 0 else 0
                    except:
                        new_streak = 1
                elif xp_gained > 0:
                    new_streak = 1

                cursor.execute('UPDATE [user] SET xp_points = ?, daily_streak = ?, last_practice_date = ? WHERE id = ?',
                               (new_xp, new_streak, datetime.datetime.utcnow(), session['user_id']))

            # Check if progress exists
            cursor.execute('SELECT id, attempts, accuracy FROM user_progress WHERE user_id = ? AND lesson_id = ?', 
                           (session['user_id'], lesson_id))
            row = cursor.fetchone()
            
            ts = datetime.datetime.utcnow().isoformat()
            if row:
                pid, att, acc = row
                new_att = att + 1
                new_acc = max(acc, confidence if is_correct else 0)
                status = 'completed' if is_correct or row[0] == 'completed' else 'in_progress'
                cursor.execute('UPDATE user_progress SET status = ?, accuracy = ?, attempts = ?, last_practice = ? WHERE id = ?',
                               (status, new_acc, new_att, ts, pid))
            else:
                status = 'completed' if is_correct else 'in_progress'
                cursor.execute('INSERT INTO user_progress (user_id, lesson_id, status, accuracy, attempts, last_practice) VALUES (?, ?, ?, ?, ?, ?)',
                               (session['user_id'], lesson_id, status, confidence if is_correct else 0, 1, ts))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating progress: {e}")

    return jsonify({
        'success': True,
        'is_correct': is_correct,
        'feedback': feedback,
        'target': target,
        'accuracy': confidence
    })

@app.route('/api/ai/progress')
def api_get_user_progress():
    """Get all progress for the logged-in user."""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        # Get user XP and Streak
        cursor.execute('SELECT xp_points, daily_streak FROM [user] WHERE id = ?', (session['user_id'],))
        u_row = cursor.fetchone()
        xp, streak = (u_row[0], u_row[1]) if u_row else (0, 0)

        cursor.execute('SELECT lesson_id, status, accuracy, attempts, last_practice FROM user_progress WHERE user_id = ?', 
                       (session['user_id'],))
        rows = cursor.fetchall()
        conn.close()
        
        progress = []
        for r in rows:
            progress.append({
                'lesson_id': r[0],
                'status': r[1],
                'accuracy': r[2],
                'attempts': r[3],
                'last_practice': r[4]
            })
        return jsonify({'success': True, 'progress': progress, 'xp': xp, 'streak': streak})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/learn')
def learn():
    """AI Tutor landing — shows lesson picker and starts a lesson (MVP)."""
    user = get_logged_in_user()
    return render_template('learn.html', user=user)


@app.route('/practice')
def practice():
    """Practice mode — select signs and start adaptive practice (MVP)."""
    user = get_logged_in_user()
    return render_template('practice.html', user=user)


@app.route('/revision')
def revision():
    """Revision / mistake review page (MVP)."""
    user = get_logged_in_user()
    return render_template('revision.html', user=user)


@app.route('/api/ai/next-lesson')
def api_next_lesson():
    """Placeholder API to return the next lesson for the user. Returns JSON."""
    # In MVP, return a static lesson; later this should be personalized
    lesson = {
        'id': 'lesson_1',
        'title': 'Alphabet: A',
        'steps': [
            {'type': 'video', 'src': '/static/img/sample_sign.mp4', 'caption': 'Watch the demo'},
            {'type': 'instruction', 'text': 'Hold hand in A position'},
            {'type': 'practice', 'prompt': 'Try signing A and press Check'}
        ]
    }
    return jsonify({'success': True, 'lesson': lesson})

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')


@app.route('/sign-basics')
def sign_basics():
    return render_template('sign_basics.html')


@app.route('/supported-languages')
def supported_languages():
    return render_template('supported_languages.html')


@app.route('/faqs')
def faqs():
    return render_template('faqs.html')


@app.route('/dataset-info')
def dataset_info():
    return render_template('dataset_info.html')


@app.route('/api-access')
def api_access():
    return render_template('api_access.html')


@app.route('/services')
def services():
    # simple services list page
    return render_template('services.html')


@app.route('/projects')
def projects():
    return render_template('projects.html')


@app.route('/projects/research')
def research_projects():
    return render_template('research_projects.html')


@app.route('/projects/student-projects')
def student_projects():
    return render_template('student_projects.html')


@app.route('/projects/future-roadmap')
def future_roadmap():
    return render_template('future_roadmap.html')


@app.route('/news')
def news():
    return render_template('news.html')


@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


# Serve images requested under /static/img/... from top-level `img/` folder when missing in `static/img`
@app.route('/static/img/<path:filename>')
def static_img_fallback(filename):
    # prefer files under static/img if present
    static_path = os.path.join(app.root_path, 'static', 'img', filename)
    if os.path.exists(static_path):
        return send_from_directory(os.path.join(app.root_path, 'static', 'img'), filename)
    # fallback to top-level img/ directory
    alt_dir = os.path.join(app.root_path, 'img')
    alt_path = os.path.join(alt_dir, filename)
    if os.path.exists(alt_path):
        return send_from_directory(alt_dir, filename)
    # not found
    abort(404)


    # If an external AI key is configured, prefer using it for richer replies.
    ai_key = app.config.get('AI_KEY') or os.environ.get('AI_KEY') or os.environ.get('OPENAI_API_KEY')
    ai_model = app.config.get('AI_MODEL', 'gpt-3.5-turbo')
    if ai_key:
        try:
            headers = {'Authorization': f'Bearer {ai_key}', 'Content-Type': 'application/json'}
            payload = {
                'model': ai_model,
                'messages': [
                    {'role': 'system', 'content': 'You are SynHand assistant. Provide concise troubleshooting help.'},
                    {'role': 'user', 'content': q}
                ],
                'temperature': 0.2,
                'max_tokens': 300,
            }
            resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=10)
            if resp.ok:
                data = resp.json()
                # Extract reply from response (safe access)
                ans = None
                if isinstance(data, dict):
                    choices = data.get('choices') or []
                    if choices and isinstance(choices, list):
                        msg = choices[0].get('message') or {}
                        ans = msg.get('content')
                if ans:
                    return jsonify({'answer': ans})
        except Exception:
            # silently fall back to canned replies on any error
            pass



# API: check username availability
@app.route('/api/check-username', methods=['GET'])
def api_check_username():
    username = request.args.get('username', '').strip()
    if not username:
        return jsonify({'available': False})
    available = not check_username_exists(app.config['DATABASE'], username)
    return jsonify({'available': available})

# API: check email availability
@app.route('/api/check-email', methods=['GET'])
def api_check_email():
    email = request.args.get('email', '').strip()
    if not email:
        return jsonify({'available': False})
    available = not check_email_exists(app.config['DATABASE'], email)
    return jsonify({'available': available})

# API Auth routes
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    identifier = data.get('identifier')
    password = data.get('password')
    
    if not identifier or not password:
        return jsonify({'success': False, 'message': 'Identifier and password are required'})
    
    # Try the new users table first
    user = verify_user_in_users(app.config['DATABASE'], identifier, password)
    
    # Fallback to the old [user] table
    if not user:
        user = verify_user(app.config['DATABASE'], identifier, password)
        
    if user:
        session['user_id'] = user['id']
        # Update last login
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            if user.get('_table') == 'users':
                cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
            else:
                cursor.execute('UPDATE [user] SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'})

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')
    
    if not all([name, email, username, password]):
        return jsonify({'success': False, 'message': 'All fields are required'})
    
    # Check if username or email already exists in any table
    if check_username_exists(app.config['DATABASE'], username):
        return jsonify({'success': False, 'message': 'Username already taken.'})
    if check_email_exists(app.config['DATABASE'], email):
        return jsonify({'success': False, 'message': 'Email already registered.'})
    
    try:
        user_id = create_user_in_users(app.config['DATABASE'], name, email, username, password)
        return jsonify({'success': True, 'message': 'Registration successful'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Auth routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    # If already logged in, greet and redirect to profile
    if 'user_id' in session:
        user = get_logged_in_user()
        if user:
            flash(f"Welcome back, {user.get('first_name') or user.get('username') or user.get('name')}", 'info')
            return redirect(url_for('profile'))
        else:
            session.pop('user_id', None)

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = verify_user(app.config['DATABASE'], username, password)
        if user:
            session['user_id'] = user['id']
            # Update last login
            try:
                conn = get_db_connection(app.config['DATABASE'])
                cursor = conn.cursor()
                if user.get('_table') == 'users':
                    cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
                else:
                    cursor.execute('UPDATE [user] SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
                conn.commit()
                conn.close()
            except Exception:
                pass
            flash('You successfully logged in', 'success')
            return redirect(url_for('profile'))
        else:
            flash('Invalid credentials.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # New registration flow: render form and create account
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        dob = request.form.get('dob')
        gender = request.form.get('gender')
        username = (request.form.get('username') or request.form.get('email') or '').strip().lower()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password')
        password2 = request.form.get('password2')
        # Basic validation
        if not username or not email or not password:
            flash('Please fill all required fields.', 'error')
            return render_template('login.html', active_tab='register')
        if password != password2:
            flash('Passwords do not match.', 'error')
            return render_template('login.html', active_tab='register')
        try:
            # ensure username/email uniqueness across both tables
            if check_username_exists(app.config['DATABASE'], username):
                flash('Username already taken.', 'error')
                return render_template('login.html', active_tab='register')
            if check_email_exists(app.config['DATABASE'], email):
                flash('Email already registered.', 'error')
                return render_template('login.html', active_tab='register')

            # server-side validation: basic checks
            if len(password) < 8:
                flash('Password must be at least 8 characters.', 'error')
                return render_template('login.html', active_tab='register')

            # create a confirmation token and save it
            token = serializer.dumps(email, salt='email-confirm')
            user_id = create_user(app.config['DATABASE'], username, password, email, first_name, last_name, dob, gender, is_confirmed=False, confirm_token=token)

            # send confirmation email (best-effort)
            try:
                msg = EmailMessage()
                confirm_url = url_for('confirm_email', token=token, _external=True)
                msg['Subject'] = 'Confirm your SynHand account'
                msg['From'] = app.config['MAIL_DEFAULT_SENDER']
                msg['To'] = email
                msg.set_content(f"Hi {first_name or username},\n\nPlease confirm your account by visiting: {confirm_url}\n\nIf you didn't request this, ignore this email.")

                with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as s:
                    if app.config['MAIL_USE_TLS']:
                        s.starttls()
                    if app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']:
                        s.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
                    s.send_message(msg)
            except Exception:
                # Don't block registration if email fails; just flash a notice
                flash('Account created but confirmation email could not be sent. Please contact support.', 'warning')

            flash('Account created successfully. Please log in with your credentials.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(str(e), 'error')
            return render_template('login.html', active_tab='register')
    # Render the combined auth page with the register tab active
    return render_template('login.html', active_tab='register')


@app.route('/confirm/<token>')
def confirm_email(token):
    try:
        email = serializer.loads(token, salt='email-confirm', max_age=60*60*24)
    except SignatureExpired:
        flash('Confirmation link expired. Please request a new confirmation email.', 'error')
        return redirect(url_for('login'))
    except BadSignature:
        flash('Invalid confirmation token.', 'error')
        return redirect(url_for('login'))

    user = get_user_by_email(app.config['DATABASE'], email)
    if not user:
        flash('Account not found.', 'error')
        return redirect(url_for('register'))

    # mark confirmed in DB
    conn = None
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('UPDATE [user] SET is_confirmed = 1, confirm_token = NULL WHERE email = ?', (email,))
        conn.commit()
        flash('Email confirmed. Thank you!', 'success')
    finally:
        if 'conn' in locals() and conn:
            conn.close()
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = get_user_by_email(app.config['DATABASE'], email)
        if not user:
            flash('Email address not found.', 'error')
            return render_template('forgot_password.html')
        
        # Generate token
        import secrets
        token = secrets.token_urlsafe(32)
        expiry = (datetime.datetime.now() + datetime.timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute('UPDATE [user] SET reset_token = ?, reset_expiry = ? WHERE email = ?', (token, expiry, email))
            conn.commit()
            conn.close()
            
            # Send email
            msg = EmailMessage()
            reset_url = url_for('reset_password', token=token, _external=True)
            msg['Subject'] = 'Reset your Suhwa password'
            msg['From'] = app.config['MAIL_DEFAULT_SENDER']
            msg['To'] = email
            msg.set_content(f"Hi,\n\nYou requested a password reset. Click the link below to set a new password:\n\n{reset_url}\n\nThis link will expire in 15 minutes. If you didn't request this, please ignore this email.")
            
            with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as s:
                if app.config['MAIL_USE_TLS']:
                    s.starttls()
                if app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']:
                    s.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
                s.send_message(msg)
            
            flash('Password reset link sent to your email.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return render_template('forgot_password.html')
            
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # Validate token
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Check token and expiry
        cursor.execute('SELECT id, email FROM [user] WHERE reset_token = ? AND reset_expiry > ?', (token, now))
        row = cursor.fetchone()
        user = dict_from_row(row, cursor)
        conn.close()
        
        if not user:
            flash('Invalid or expired reset link.', 'error')
            return redirect(url_for('login'))
            
        if request.method == 'POST':
            password = request.form.get('password')
            confirm = request.form.get('confirm_password')
            if not password or password != confirm:
                flash('Passwords do not match.', 'error')
                return render_template('reset_password.html')
            
            if len(password) < 8:
                flash('Password must be at least 8 characters.', 'error')
                return render_template('reset_password.html')
                
            # Update password
            pwdhash = generate_password_hash(password)
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute('UPDATE [user] SET password = ?, reset_token = NULL, reset_expiry = NULL WHERE id = ?', (pwdhash, user['id']))
            conn.commit()
            conn.close()
            
            flash('Password updated successfully. You can now log in.', 'success')
            return redirect(url_for('login'))
            
        return render_template('reset_password.html')
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/auth/firebase', methods=['POST'])
def firebase_auth():
    id_token = request.json.get('id_token')
    decoded_token = verify_firebase_token(id_token)
    if decoded_token:
        firebase_uid = decoded_token['uid']
        email = decoded_token.get('email')
        name = decoded_token.get('name')
        user = create_or_get_firebase_user(app.config['DATABASE'], firebase_uid, email, name)
        session['user_id'] = user['id']
        return jsonify({'success': True, 'user': user})
    return jsonify({'success': False}), 401

@app.route('/camera_feed')
def camera_feed():
    def generate():
        cam = get_camera()
        while True:
            with camera_lock:
                success, frame = cam.read()
                if not success:
                    break
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/yolo_feed')
def yolo_feed():
    def generate():
        cam = get_camera()
        while True:
            with camera_lock:
                success, frame = cam.read()
                if not success:
                    break
                
                # Perform YOLO inference
                if MODEL_YOLO is not None:
                    try:
                        # YOLO results
                        # Use .model attribute of YOLOPredictor
                        results = MODEL_YOLO.model(frame)  # type: ignore
                        res = results[0]
                        annotated_frame = res.plot()
                        display_frame = annotated_frame
                        
                        # Extra logic for action recognition
                        if hasattr(res, 'boxes') and len(res.boxes) > 0:
                            top_box = res.boxes[0]
                            label = res.names[int(top_box.cls[0])]
                            conf = float(top_box.conf[0])
                            socketio.emit('yolo_prediction', {'label': label, 'confidence': conf})
                    except Exception as e:
                        print(f"YOLO Feed Error: {e}")
                        display_frame = frame
                elif ROBOFLOW_CLIENT is not None:
                    try:
                        # Fallback to Roboflow Inference Workflow if local model is not available
                        temp_image_path = os.path.join(app.root_path, 'temp_yolo_feed.jpg')
                        cv2.imwrite(temp_image_path, frame)
                        
                        workflow_result = ROBOFLOW_CLIENT.run_workflow(
                            workspace_name=ROBOFLOW_WORKSPACE,
                            workflow_id=ROBOFLOW_WORKFLOW_ID,
                            images={"image": temp_image_path},
                            parameters={"classes": "A, B, C, D, E, F, G, H, I, J"},
                            use_cache=True
                        )
                        
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                            
                        display_frame = frame.copy()
                        predictions = []
                        if isinstance(workflow_result, list) and len(workflow_result) > 0:
                            output = workflow_result[0].get('outputs', {})
                            for key, value in output.items():
                                if isinstance(value, list) and len(value) > 0 and 'class_name' in value[0]:
                                    predictions = value
                                    break
                        
                        for pred in predictions:
                            # Draw bounding boxes if coordinates are available
                            # Note: coordinate names might vary (x, y, width, height OR x1, y1, x2, y2)
                            x = pred.get('x')
                            y = pred.get('y')
                            w = pred.get('width')
                            h = pred.get('height')
                            label = pred.get('class_name')
                            conf = pred.get('confidence', 0.0)
                            
                            if x is not None and y is not None and w is not None and h is not None:
                                x1, y1 = int(x - w/2), int(y - h/2)
                                x2, y2 = int(x + w/2), int(y + h/2)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                        if predictions:
                            top_pred = max(predictions, key=lambda x: x.get('confidence', 0))
                            socketio.emit('yolo_prediction', {
                                'label': top_pred.get('class_name'),
                                'confidence': top_pred.get('confidence', 0.0)
                            })
                    except Exception as e:
                        print(f"Roboflow Feed Error: {e}")
                        display_frame = frame
                else:
                    display_frame = frame

                ret, buffer = cv2.imencode('.jpg', display_frame)
                frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05) # Slightly faster for YOLO
            
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/yolo_detect')
def yolo_detect():
    user = get_logged_in_user()
    return render_template('yolo_detect.html', user=user)



@app.route('/api/save-landmark', methods=['POST'])
def api_save_landmark():
    """Save a single labeled landmark vector to dataset/landmarks.csv"""
    data = request.get_json() or {}
    label = data.get('label')
    landmarks = data.get('landmarks')
    if not label or not landmarks:
        return jsonify({'success': False, 'error': 'label and landmarks required'}), 400
    # ensure dataset dir exists
    ds_dir = os.path.join(app.root_path, 'dataset')
    os.makedirs(ds_dir, exist_ok=True)
    csv_path = os.path.join(ds_dir, 'landmarks.csv')
    try:
        import csv
        # create file with header if not exists
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf8') as fh:
            writer = csv.writer(fh)
            if write_header:
                header = ['label', 'timestamp'] + [f'l{i}' for i in range(len(landmarks))]
                writer.writerow(header)
            row = [label, datetime.datetime.utcnow().isoformat()] + landmarks
            writer.writerow(row)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/save-landmark-batch', methods=['POST'])
def api_save_landmark_batch():
    """Save a batch of landmarks as a sequence numpy file under dataset/sequences."""
    data = request.get_json() or {}
    label = data.get('label')
    frames = data.get('frames')  # list of landmark lists
    if not label or not frames or not isinstance(frames, list):
        return jsonify({'success': False, 'error': 'label and frames[] required'}), 400
    seq_dir = os.path.join(app.root_path, 'dataset', 'sequences')
    os.makedirs(seq_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    fname = f"{label}_{ts}.npy"
    path = os.path.join(seq_dir, fname)
    try:
        import numpy as np
        arr = np.array(frames, dtype=np.float32)
        np.save(path, arr)
        return jsonify({'success': True, 'path': os.path.relpath(path, app.root_path)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/collect')
def collect_page():
    return render_template('collect.html')

@app.route('/api/predict-image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    try:
        image_bytes = file.read()
        # read optional language param from form
        lang = request.form.get('lang') or request.args.get('lang') or 'asl'
        # Allow a dedicated image model path; prefer preloaded per-language image model
        img_model_entry = IMAGE_MODELS.get(lang) if 'IMAGE_MODELS' in globals() else None
        if lang == 'mnist' and MODEL_MNIST is not None:
            result = predict_from_image_bytes_cnn(image_bytes, model=MODEL_MNIST)
        elif img_model_entry and img_model_entry.get('model') is not None:
            result = predict_from_image_bytes_cnn(image_bytes, model=img_model_entry.get('model'), class_names=img_model_entry.get('class_names'))
        else:
            model_path = app.config.get(f'IMAGE_MODEL_PATH_{lang.upper()}') or app.config.get('IMAGE_MODEL_PATH') or app.config.get('MODEL_PATH')
            result = predict_from_image_bytes_cnn(image_bytes, model_path=model_path)
        # translate label/sentence according to requested lang
        raw_label = result.get('label') or result.get('sentence') or None
        subtitle_text = translate_label(raw_label, lang) if raw_label else ''
        resp = {'success': True, 'prediction': result, 'subtitle': subtitle_text, 'lang': lang}
        return jsonify(resp)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def send_contact_emails(name, email, subject, message):
    """Sends notification to admin and auto-response to user."""
    # 1. Admin Notification
    try:
        admin_msg = EmailMessage()
        admin_msg['Subject'] = f"New Contact Message: {subject}"
        admin_msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        admin_msg['To'] = app.config.get('ADMIN_EMAIL', app.config.get('MAIL_USERNAME'))  # Send to specific admin email
        
        # Plain text version
        admin_plain = f"New message from {name} ({email})\n\nSubject: {subject}\n\nMessage:\n{message}"
        admin_msg.set_content(admin_plain)

        # HTML version for colorful admin alert
        admin_html = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden;">
                    <div style="background-color: #F5A623; padding: 20px; text-align: center;">
                        <h1 style="color: #ffffff; margin: 0;">New Contact Alert</h1>
                        <p style="color: #fff0d0; margin: 5px 0 0 0;">Suhwa Sign Language App</p>
                    </div>
                    <div style="padding: 30px;">
                        <h2 style="color: #F5A623; border-bottom: 2px solid #F5A623; padding-bottom: 10px;">Message Details</h2>
                        <table style="width: 100%; border-collapse: collapse; margin-top: 20px;">
                            <tr>
                                <td style="padding: 10px; font-weight: bold; width: 30%; border-bottom: 1px solid #eee;">Name:</td>
                                <td style="padding: 10px; border-bottom: 1px solid #eee;">{name}</td>
                            </tr>
                            <tr>
                                <td style="padding: 10px; font-weight: bold; border-bottom: 1px solid #eee;">Email:</td>
                                <td style="padding: 10px; border-bottom: 1px solid #eee;">{email}</td>
                            </tr>
                            <tr>
                                <td style="padding: 10px; font-weight: bold; border-bottom: 1px solid #eee;">Subject:</td>
                                <td style="padding: 10px; border-bottom: 1px solid #eee;">{subject}</td>
                            </tr>
                        </table>
                        <div style="margin-top: 25px; padding: 20px; background-color: #fff9f0; border-radius: 5px; border: 1px solid #ffe0b0;">
                            <h3 style="margin-top: 0; color: #d35400;">Message:</h3>
                            <p style="margin-bottom: 0; white-space: pre-wrap;">{message}</p>
                        </div>
                    </div>
                    <div style="background-color: #f4f4f4; padding: 15px; text-align: center; font-size: 12px; color: #777;">
                        <p style="margin: 0;">This is an automated notification from Suhwa App.</p>
                    </div>
                </div>
            </body>
        </html>
        """
        admin_msg.add_alternative(admin_html, subtype='html')

        # 2. User Auto-Response
        user_msg = EmailMessage()
        user_msg['Subject'] = "Thank you for contacting Sign Language Detection App"
        user_msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        user_msg['To'] = email
        
        # Plain text version
        plain_content = f"Hello {name},\n\nThank you for reaching out to the Sign Language Detection App team. We have received your message and our team will get back to you shortly.\n\nIf your query is urgent, please reply to this email.\n\nBest regards,\nSign Language Detection App Team\nAccessibility • Inclusion • Innovation"
        user_msg.set_content(plain_content)

        # HTML version for "colorful" look
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden;">
                    <div style="background-color: #4A90E2; padding: 20px; text-align: center;">
                        <h1 style="color: #ffffff; margin: 0;">Suhwa</h1>
                        <p style="color: #e0f0ff; margin: 5px 0 0 0;">Sign Language Detection App</p>
                    </div>
                    <div style="padding: 30px;">
                        <h2 style="color: #4A90E2;">Hello {name},</h2>
                        <p>Thank you for reaching out to us! We've received your message and our team is already looking into it.</p>
                        <p>We'll get back to you as soon as possible.</p>
                        <div style="background-color: #f9f9f9; border-left: 4px solid #4A90E2; padding: 15px; margin: 20px 0;">
                            <p style="margin: 0; font-style: italic;">"Breaking barriers through technology and inclusion."</p>
                        </div>
                        <p>If your query is urgent, simply reply to this email.</p>
                        <br>
                        <p style="margin-bottom: 0;">Best regards,</p>
                        <p style="margin-top: 5px; font-weight: bold; color: #4A90E2;">The Suhwa Team</p>
                    </div>
                    <div style="background-color: #f4f4f4; padding: 15px; text-align: center; font-size: 12px; color: #777;">
                        <p style="margin: 0;">Accessibility • Inclusion • Innovation</p>
                    </div>
                </div>
            </body>
        </html>
        """
        user_msg.add_alternative(html_content, subtype='html')

        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as s:
            if app.config['MAIL_USE_TLS']:
                s.starttls()
            if app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']:
                s.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            s.send_message(admin_msg)
            s.send_message(user_msg)
        return True
    except Exception as e:
        print(f"Error sending emails: {e}")
        return False

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        if not all([name, email, message]):
            flash('Name, email and message are required.', 'error')
            return render_template('contact.html')

        # Store in DB
        try:
            conn = get_db_connection(app.config['DATABASE'])
            cursor = conn.cursor()
            cursor.execute('INSERT INTO contact_messages (name, email, subject, message) VALUES (?, ?, ?, ?)',
                           (name, email, subject, message))
            conn.commit()
            conn.close()
        except Exception as e:
            flash(f'Database error: {str(e)}', 'error')
            return render_template('contact.html')

        # Send Emails
        send_contact_emails(name, email, subject, message)

        flash('Your message has been sent successfully. Please check your email.', 'success')
        return redirect(url_for('contact'))

    return render_template('contact.html')

@app.route('/api/contact', methods=['POST'])
def api_contact():
    data = request.get_json() or {}
    name = data.get('name')
    email = data.get('email')
    subject = data.get('subject')
    message = data.get('message')

    if not all([name, email, message]):
        return jsonify({'status': 'error', 'message': 'Name, email and message are required'}), 400

    # Store in DB
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('INSERT INTO contact_messages (name, email, subject, message) VALUES (?, ?, ?, ?)',
                       (name, email, subject, message))
        conn.commit()
        conn.close()
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Database error: {str(e)}'}), 500

    # Send Emails
    send_contact_emails(name, email, subject, message)

    return jsonify({'status': 'success', 'message': 'Your message has been sent successfully'})

@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.get_json() or {}
    email = data.get('email')

    if not email:
        return jsonify({'status': 'error', 'message': 'Email is required'}), 400

    # Basic email validation
    import re
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({'status': 'error', 'message': 'Invalid email address'}), 400

    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('INSERT INTO newsletter_subscribers (email) VALUES (?)', (email,))
        conn.commit()
        conn.close()
    except Exception as e:
        # Check if it's a duplicate entry error
        error_msg = str(e).lower()
        if 'unique' in error_msg or 'duplicate' in error_msg:
            return jsonify({'status': 'error', 'message': 'You are already subscribed!'}), 409
        return jsonify({'status': 'error', 'message': f'Database error: {str(e)}'}), 500

    # Send Welcome Email
    try:
        msg = EmailMessage()
        msg['Subject'] = "Welcome to Suhwa Newsletter"
        msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        msg['To'] = email
        
        # Plain text
        msg.set_content(f"Hi,\n\nThank you for subscribing to Suhwa's newsletter! You'll now receive the latest updates on AI sign language advancements.\n\nBest regards,\nThe Suhwa Team")
        
        # Colorful HTML
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden;">
                    <div style="background-color: #4A90E2; padding: 20px; text-align: center;">
                        <h1 style="color: #ffffff; margin: 0;">Welcome to Suhwa!</h1>
                        <p style="color: #e0f0ff; margin: 5px 0 0 0;">Newsletter Subscription</p>
                    </div>
                    <div style="padding: 30px;">
                        <h2 style="color: #4A90E2;">You're on the list!</h2>
                        <p>Thank you for subscribing to the Suhwa newsletter. We're excited to have you with us as we bridge communication gaps with AI.</p>
                        <p>You'll receive updates on:</p>
                        <ul style="list-style-type: none; padding-left: 0;">
                            <li><i style="color: #4A90E2; margin-right: 10px;">✔</i> New Sign Detection Features</li>
                            <li><i style="color: #4A90E2; margin-right: 10px;">✔</i> Educational Content & Tutorials</li>
                            <li><i style="color: #4A90E2; margin-right: 10px;">✔</i> Accessibility News</li>
                        </ul>
                        <br>
                        <p>Stay tuned for more updates!</p>
                        <br>
                        <p style="margin-bottom: 0;">Best regards,</p>
                        <p style="margin-top: 5px; font-weight: bold; color: #4A90E2;">The Suhwa Team</p>
                    </div>
                    <div style="background-color: #f4f4f4; padding: 15px; text-align: center; font-size: 12px; color: #777;">
                        <p style="margin: 0;">Accessibility • Inclusion • Innovation</p>
                    </div>
                </div>
            </body>
        </html>
        """
        msg.add_alternative(html_content, subtype='html')

        with smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT']) as s:
            if app.config['MAIL_USE_TLS']:
                s.starttls()
            s.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            s.send_message(msg)
    except Exception as e:
        print(f"Newsletter email error: {e}")

    return jsonify({'status': 'success', 'message': 'Thank you for subscribing!'})

@app.route('/meet')
def create_meeting():
    room_id = str(uuid.uuid4())[:8]
    return redirect(url_for('join_meeting', room_id=room_id))

@app.route('/meet/<room_id>')
def join_meeting(room_id):
    user = get_logged_in_user()
    return render_template('video_call.html', room_id=room_id, user=user)

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    sid = getattr(request, 'sid', None)
    emit('user-joined', {'userId': sid}, to=room, include_self=False)

@socketio.on('offer')
def on_offer(data):
    data['userId'] = getattr(request, 'sid', None)
    emit('offer', data, to=data['room'], include_self=False)

@socketio.on('answer')
def on_answer(data):
    data['userId'] = getattr(request, 'sid', None)
    emit('answer', data, to=data['room'], include_self=False)

@socketio.on('ice-candidate')
def on_ice_candidate(data):
    data['userId'] = getattr(request, 'sid', None)
    emit('ice-candidate', data, to=data['room'], include_self=False)

@socketio.on('subtitle')
def on_subtitle(data):
    # data should contain: room, text, type (speech/sign), userId
    emit('subtitle', data, to=data['room'], include_self=False)

from functools import wraps

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash("Please log in to access the admin panel.", "warning")
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == app.config['ADMIN_USERNAME'] and password == app.config['ADMIN_PASSWORD']:
            session['admin_logged_in'] = True
            flash("Logged in successfully as Admin", "success")
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid admin credentials", "danger")
            
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash("Logged out from admin panel", "info")
    return redirect(url_for('admin_login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        # Get all table names
        if "DRIVER=" in app.config['DATABASE']:
            cursor.execute("SELECT name FROM sysobjects WHERE xtype='U'")
        else:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get stats for dashboard
        stats = {}
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                result = cursor.fetchone()
                stats[table] = result[0] if result else 0
            except:
                stats[table] = 0
                
        conn.close()
        return render_template('admin_panel.html', 
                               tables=tables, 
                               stats=stats, 
                               active_table=None,
                               columns=[],
                               rows=[],
                               rows_json='[]')
    except Exception as e:
        flash(f"Error accessing admin dashboard: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/admin/table/<table_name>')
@admin_required
def admin_table(table_name):
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        # Get all table names for sidebar
        if "DRIVER=" in app.config['DATABASE']:
            cursor.execute("SELECT name FROM sysobjects WHERE xtype='U'")
        else:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get table data
        cursor.execute(f"SELECT * FROM [{table_name}]")
        columns = [column[0] for column in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return render_template('admin_panel.html', 
                               tables=tables, 
                               active_table=table_name, 
                               columns=columns, 
                               rows=rows,
                               rows_json=json.dumps(rows, default=str))
    except Exception as e:
        flash(f"Error accessing table {table_name}: {e}", "danger")
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/update/<table_name>', methods=['POST'])
@admin_required
def admin_update(table_name):
    try:
        data = request.form.to_dict()
        row_id = data.get('id')
        if not row_id:
            flash("Missing ID for update", "danger")
            return redirect(url_for('admin_table', table_name=table_name))
        
        # Remove ID from data to update
        update_data = {k: v for k, v in data.items() if k != 'id'}
        
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        # Construct update query
        placeholders = ", ".join([f"[{k}] = ?" for k in update_data.keys()])
        values = list(update_data.values())
        values.append(row_id)
        
        query = f"UPDATE [{table_name}] SET {placeholders} WHERE id = ?"
        cursor.execute(query, values)
        
        conn.commit()
        conn.close()
        flash(f"Record in {table_name} updated successfully", "success")
    except Exception as e:
        flash(f"Error updating record: {e}", "danger")
    
    return redirect(url_for('admin_table', table_name=table_name))

@app.route('/admin/delete/<table_name>/<int:row_id>', methods=['POST'])
@admin_required
def admin_delete(table_name, row_id):
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM [{table_name}] WHERE id = ?", (row_id,))
        conn.commit()
        conn.close()
        flash(f"Record in {table_name} deleted successfully", "success")
    except Exception as e:
        flash(f"Error deleting record: {e}", "danger")
        
    return redirect(url_for('admin_table', table_name=table_name))

@app.route('/api/encryption/generate-keys', methods=['POST'])
def generate_encryption_keys():
    """Generate encryption keys for current user"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        from backend.utils.crypto import crypto
        
        user_id = session['user_id']
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_encryption_keys WHERE user_id = ?', (user_id,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('SELECT public_key FROM user_encryption_keys WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            conn.close()
            return jsonify({
                'success': True,
                'public_key': result[0] if result else None,
                'message': 'Keys already exist'
            }), 200
        
        public_key, private_key = crypto.generate_key_pair()
        
        cursor.execute('''INSERT INTO user_encryption_keys (user_id, public_key, private_key, key_algorithm)
                         VALUES (?, ?, ?, ?)''',
                      (user_id, public_key, private_key, 'RSA-2048'))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'public_key': public_key,
            'message': 'Keys generated successfully'
        }), 201
    except Exception as e:
        print(f"Error generating keys: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/encryption/get-public-key/<int:user_id>', methods=['GET'])
def get_user_public_key(user_id):
    """Get public key of a specific user"""
    try:
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('SELECT public_key FROM user_encryption_keys WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return jsonify({
                'success': True,
                'public_key': result[0]
            }), 200
        else:
            return jsonify({'success': False, 'error': 'User keys not found'}), 404
    except Exception as e:
        print(f"Error getting public key: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/encryption/encrypt-message', methods=['POST'])
def encrypt_message():
    """Encrypt a message for a recipient"""
    try:
        data = request.json
        message = data.get('message')
        recipient_id = data.get('recipient_id')
        
        if not message or not recipient_id:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        from backend.utils.crypto import message_encryption
        
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('SELECT public_key FROM user_encryption_keys WHERE user_id = ?', (recipient_id,))
        recipient_result = cursor.fetchone()
        
        cursor.execute('SELECT private_key FROM user_encryption_keys WHERE user_id = ?', (session['user_id'],))
        sender_result = cursor.fetchone()
        
        conn.close()
        
        if not recipient_result or not sender_result:
            return jsonify({'success': False, 'error': 'Encryption keys not found'}), 404
        
        encrypted_data = message_encryption.encrypt_for_recipient(
            message,
            recipient_result[0],
            sender_result[0]
        )
        
        return jsonify({
            'success': True,
            'encrypted': encrypted_data['encrypted'],
            'signature': encrypted_data['signature'],
            'algorithm': encrypted_data['algorithm']
        }), 200
    except Exception as e:
        print(f"Error encrypting message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/encryption/decrypt-message', methods=['POST'])
def decrypt_message_api():
    """Decrypt a message"""
    try:
        data = request.json
        encrypted_message = data.get('encrypted')
        signature = data.get('signature')
        sender_id = data.get('sender_id')
        
        if not encrypted_message or not sender_id:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401
        
        from backend.utils.crypto import message_encryption
        
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('SELECT private_key FROM user_encryption_keys WHERE user_id = ?', (session['user_id'],))
        recipient_result = cursor.fetchone()
        
        cursor.execute('SELECT public_key FROM user_encryption_keys WHERE user_id = ?', (sender_id,))
        sender_result = cursor.fetchone()
        
        conn.close()
        
        if not recipient_result or not sender_result:
            return jsonify({'success': False, 'error': 'Encryption keys not found'}), 404
        
        encrypted_data = {
            'encrypted': encrypted_message,
            'signature': signature
        }
        
        is_valid, decrypted = message_encryption.decrypt_message(
            encrypted_data,
            recipient_result[0],
            sender_result[0]
        )
        
        return jsonify({
            'success': True,
            'decrypted': decrypted,
            'is_signature_valid': is_valid
        }), 200
    except Exception as e:
        print(f"Error decrypting message: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@socketio.on('send_encrypted_message')
def handle_encrypted_message(data):
    """Handle encrypted messages in chat rooms"""
    try:
        room_id = data.get('room_id')
        user_id = data.get('user_id')
        username = data.get('username')
        encrypted_message = data.get('encrypted_message')
        signature = data.get('signature')
        
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''INSERT INTO messages (room_id, user_id, message_text, message_type, 
                         is_encrypted, encryption_algorithm, message_signature, timestamp)
                         VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                      (room_id, user_id, encrypted_message, 'text', 1, 'RSA-OAEP-SHA256', signature))
        conn.commit()
        
        msg_id = cursor.lastrowid  # type: ignore
        timestamp = datetime.datetime.now().isoformat()
        
        conn.close()
        
        emit('new_encrypted_message', {
            'id': msg_id,
            'room_id': room_id,
            'user_id': user_id,
            'username': username,
            'encrypted_message': encrypted_message,
            'signature': signature,
            'timestamp': timestamp,
            'is_encrypted': True
        }, to=f'room_{room_id}')
    except Exception as e:
        print(f"Error handling encrypted message: {e}")

@socketio.on('send_encrypted_private_message')
def handle_encrypted_private_message(data):
    """Handle encrypted private messages"""
    try:
        receiver_id = data.get('receiver_id')
        encrypted_message = data.get('encrypted_message')
        signature = data.get('signature')
        
        if 'user_id' not in session:
            emit('error', {'message': 'Not authenticated'})
            return
        
        conn = get_db_connection(app.config['DATABASE'])
        cursor = conn.cursor()
        
        cursor.execute('''INSERT INTO private_messages (sender_id, receiver_id, message_text, 
                         is_encrypted, encryption_algorithm, message_signature)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                      (session['user_id'], receiver_id, encrypted_message, 1, 'RSA-OAEP-SHA256', signature))
        conn.commit()
        
        user = get_logged_in_user()
        timestamp = datetime.datetime.now().isoformat()
        
        conn.close()
        
        username = (user.get('username') or user.get('name')) if user else 'Unknown User'
        
        emit('new_encrypted_private_message', {
            'sender_id': session['user_id'],
            'receiver_id': receiver_id,
            'username': username,
            'encrypted_message': encrypted_message,
            'signature': signature,
            'timestamp': timestamp,
            'is_encrypted': True
        })
    except Exception as e:
        print(f"Error handling encrypted private message: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True)

