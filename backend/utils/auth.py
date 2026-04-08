import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from backend.utils.db import get_db_connection, dict_from_row

def get_user_by_username(database, username):
    conn = get_db_connection(database)
    cursor = conn.cursor()
    # case-insensitive username lookup
    cursor.execute('SELECT id, username, password, email, first_name, last_name, dob, gender, is_confirmed, confirm_token, avatar, total_signs, last_detection, created_at, last_login, preferred_language, detection_mode, camera, output_type, hand_mode, camera_sensitivity FROM [user] WHERE LOWER(username) = LOWER(?)', (username,))
    row = cursor.fetchone()
    user = dict_from_row(row, cursor)
    conn.close()
    return user

def get_user_by_email(database, email):
    conn = get_db_connection(database)
    cursor = conn.cursor()
    # case-insensitive email lookup
    cursor.execute('SELECT id, username, password, email, first_name, last_name, dob, gender, is_confirmed, confirm_token, avatar, total_signs, last_detection, created_at, last_login, preferred_language, detection_mode, camera, output_type, hand_mode, camera_sensitivity FROM [user] WHERE LOWER(email) = LOWER(?)', (email,))
    row = cursor.fetchone()
    user = dict_from_row(row, cursor)
    conn.close()
    return user

def get_user_by_id(database, user_id):
    conn = get_db_connection(database)
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, password, email, first_name, last_name, dob, gender, is_confirmed, confirm_token, avatar, total_signs, last_detection, created_at, last_login, preferred_language, detection_mode, camera, output_type, hand_mode, camera_sensitivity FROM [user] WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    user = dict_from_row(row, cursor)
    conn.close()
    return user

def check_username_exists(database, username):
    """Check if username exists in either 'user' or 'users' tables."""
    if not username:
        return False
    # Check [user] table
    if get_user_by_username(database, username):
        return True
    # Check users table
    if get_user_from_users(database, username):
        return True
    return False

def check_email_exists(database, email):
    """Check if email exists in either 'user' or 'users' tables."""
    if not email:
        return False
    # Check [user] table
    if get_user_by_email(database, email):
        return True
    # Check users table
    if get_user_from_users(database, email):
        return True
    return False

def create_user(database, username, password, email, first_name, last_name, dob=None, gender=None, is_confirmed=False, confirm_token=None, avatar=None):
    conn = get_db_connection(database)
    cursor = conn.cursor()
    # normalize username and email to lower-case to avoid case-sensitivity issues
    username_norm = username.strip().lower() if username else ''
    email_norm = email.strip().lower() if email else ''
    hashed_password = generate_password_hash(password)
    try:
        cursor.execute('INSERT INTO [user] (username, password, email, first_name, last_name, dob, gender, is_confirmed, confirm_token, avatar, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)',
                       (username_norm, hashed_password, email_norm, first_name, last_name, dob, gender, 1 if is_confirmed else 0, confirm_token, avatar))
        user_id = None
        if "DRIVER=" in database:
             # For SQL Server, lastrowid might not work directly with all drivers
             cursor.execute("SELECT @@IDENTITY")
             user_id = cursor.fetchone()[0]
        else:
             user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except Exception as e:
        if "UNIQUE" in str(e) or "IntegrityError" in str(e):
             raise Exception('Username or email already exists')
        raise e
    finally:
        conn.close()

# New functions for 'users' table as requested
def get_user_from_users(database, identifier):
    conn = get_db_connection(database)
    cursor = conn.cursor()
    # identifier can be username or email
    cursor.execute('SELECT * FROM users WHERE LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?)', (identifier, identifier))
    row = cursor.fetchone()
    user = dict_from_row(row, cursor)
    conn.close()
    return user

def get_user_by_id_from_users(database, user_id):
    conn = get_db_connection(database)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    user = dict_from_row(row, cursor)
    conn.close()
    return user

def create_user_in_users(database, name, email, username, password):
    conn = get_db_connection(database)
    cursor = conn.cursor()
    hashed_password = generate_password_hash(password)
    try:
        cursor.execute('INSERT INTO users (name, email, username, password_hash) VALUES (?, ?, ?, ?)',
                       (name, email.lower(), username.lower(), hashed_password))
        user_id = None
        if "DRIVER=" in database:
             cursor.execute("SELECT @@IDENTITY")
             user_id = cursor.fetchone()[0]
        else:
             user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except Exception as e:
        if "UNIQUE" in str(e) or "IntegrityError" in str(e):
             raise Exception('Username or email already exists')
        raise e
    finally:
        conn.close()

def verify_user_in_users(database, identifier, password):
    user = get_user_from_users(database, identifier)
    if user and check_password_hash(user['password_hash'], password):
        user['_table'] = 'users'
        return user
    return None

def verify_user(database, username, password):
    if not username:
        return None
    username = username.strip()
    # try username lookup first (case-insensitive handled in get_user_by_username)
    user = get_user_by_username(database, username)
    # if not found, try email lookup (allow users to login with email)
    if not user:
        user = get_user_by_email(database, username)
    
    if user and check_password_hash(user['password'], password):
        user['_table'] = 'user'
        return user
        
    # fallback to 'users' table
    user = get_user_from_users(database, username)
    if user and check_password_hash(user.get('password_hash') or user.get('password'), password):
        user['_table'] = 'users'
        return user
        
    return None

def create_or_get_firebase_user(database, firebase_uid, email, name):
    conn = get_db_connection(database)
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, password, email, first_name, last_name FROM [user] WHERE firebase_uid = ?', (firebase_uid,))
    row = cursor.fetchone()
    user = dict_from_row(row, cursor)
    if user:
        conn.close()
        return user
    # Create new user
    first_name, last_name = (name.split(' ', 1) + [''])[:2]
    username = email.split('@')[0]  # simple username
    try:
        cursor.execute('INSERT INTO [user] (username, password, email, first_name, last_name, firebase_uid, created_at) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)',
                       (username, '', email, first_name, last_name, firebase_uid))
        user_id = None
        if "DRIVER=" in database:
             cursor.execute("SELECT @@IDENTITY")
             user_id = cursor.fetchone()[0]
        else:
             user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {'id': user_id, 'username': username, 'password': '', 'email': email, 'first_name': first_name, 'last_name': last_name}
    except Exception:
        if 'conn' in locals():
            conn.close()
        raise Exception('User creation failed')

def verify_firebase_token(id_token):
    # Stub for Firebase token verification
    # In real implementation, use Firebase Admin SDK
    # For now, return a mock decoded token
    return {'uid': 'mock_uid', 'email': 'mock@example.com', 'name': 'Mock User'}
