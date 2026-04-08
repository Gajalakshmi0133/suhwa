import sqlite3
import os

def get_db_connection(database_path):
    if "DRIVER=" in database_path:
        import pyodbc
        conn = pyodbc.connect(database_path)
        return conn
    else:
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        return conn

def dict_from_row(row, cursor=None):
    if row is None:
        return None
    if hasattr(row, 'keys'): # sqlite3.Row
        return dict(row)
    # pyodbc row
    if cursor:
        return {column[0]: value for column, value in zip(cursor.description, row)}
    return row # fallback

def init_db(database_path):
    try:
        if "DRIVER=" in database_path:
            import pyodbc
            conn = pyodbc.connect(database_path)
            cursor = conn.cursor()
            
            # Create users table as requested
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='users' AND xtype='U')
                CREATE TABLE users (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    name NVARCHAR(255) NOT NULL,
                    email NVARCHAR(255) UNIQUE NOT NULL,
                    username NVARCHAR(255) UNIQUE NOT NULL,
                    password_hash NVARCHAR(MAX) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # SQL Server syntax
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user' AND xtype='U')
                CREATE TABLE [user] (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    username NVARCHAR(255) UNIQUE NOT NULL,
                    password NVARCHAR(MAX) NOT NULL,
                    email NVARCHAR(255) UNIQUE NOT NULL,
                    first_name NVARCHAR(255),
                    last_name NVARCHAR(255),
                    dob NVARCHAR(50),
                    gender NVARCHAR(50),
                    is_confirmed INT DEFAULT 0,
                    confirm_token NVARCHAR(MAX),
                    firebase_uid NVARCHAR(255) UNIQUE,
                    avatar NVARCHAR(MAX),
                    preferred_language NVARCHAR(50),
                    detection_mode NVARCHAR(50),
                    camera NVARCHAR(255),
                    output_type NVARCHAR(50),
                    hand_mode NVARCHAR(50),
                    camera_sensitivity INT DEFAULT 700,
                    total_signs INT DEFAULT 0,
                    last_detection NVARCHAR(50),
                    reset_token NVARCHAR(255),
                    reset_expiry DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='detection_history' AND xtype='U')
                CREATE TABLE detection_history (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    user_id INT,
                    timestamp NVARCHAR(100),
                    text NVARCHAR(MAX),
                    confidence FLOAT,
                    raw_label NVARCHAR(255),
                    FOREIGN KEY (user_id) REFERENCES [user] (id)
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='contact_messages' AND xtype='U')
                CREATE TABLE contact_messages (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    name NVARCHAR(255) NOT NULL,
                    email NVARCHAR(255) NOT NULL,
                    subject NVARCHAR(255),
                    message NVARCHAR(MAX) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='newsletter_subscribers' AND xtype='U')
                CREATE TABLE newsletter_subscribers (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    email NVARCHAR(255) UNIQUE NOT NULL,
                    subscribed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='chat_rooms' AND xtype='U')
                CREATE TABLE chat_rooms (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    room_name NVARCHAR(255) NOT NULL,
                    description NVARCHAR(MAX),
                    room_type NVARCHAR(50) NOT NULL,
                    created_by INT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (created_by) REFERENCES [user] (id)
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='messages' AND xtype='U')
                CREATE TABLE messages (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    room_id INT NOT NULL,
                    user_id INT NOT NULL,
                    message_text NVARCHAR(MAX) NOT NULL,
                    message_type NVARCHAR(50) DEFAULT 'text',
                    audio_url NVARCHAR(MAX),
                    video_url NVARCHAR(MAX),
                    is_sign_detected INT DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (room_id) REFERENCES chat_rooms (id),
                    FOREIGN KEY (user_id) REFERENCES [user] (id)
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='private_messages' AND xtype='U')
                CREATE TABLE private_messages (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    sender_id INT NOT NULL,
                    receiver_id INT NOT NULL,
                    message_text NVARCHAR(MAX) NOT NULL,
                    is_read INT DEFAULT 0,
                    is_sign_detected INT DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sender_id) REFERENCES [user] (id),
                    FOREIGN KEY (receiver_id) REFERENCES [user] (id)
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='online_users' AND xtype='U')
                CREATE TABLE online_users (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    user_id INT NOT NULL,
                    current_room_id INT,
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES [user] (id),
                    FOREIGN KEY (current_room_id) REFERENCES chat_rooms (id)
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='room_members' AND xtype='U')
                CREATE TABLE room_members (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    room_id INT NOT NULL,
                    user_id INT NOT NULL,
                    joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (room_id) REFERENCES chat_rooms (id),
                    FOREIGN KEY (user_id) REFERENCES [user] (id),
                    UNIQUE(room_id, user_id)
                )
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='messages' AND COLUMN_NAME='message_type')
                ALTER TABLE messages ADD message_type NVARCHAR(50) DEFAULT 'text'
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='messages' AND COLUMN_NAME='audio_url')
                ALTER TABLE messages ADD audio_url NVARCHAR(MAX)
            ''')

            cursor.execute('''
                IF NOT EXISTS (SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='messages' AND COLUMN_NAME='video_url')
                ALTER TABLE messages ADD video_url NVARCHAR(MAX)
            ''')
        else:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()

            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create user table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS [user] (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    dob TEXT,
                    gender TEXT,
                    is_confirmed INTEGER DEFAULT 0,
                    confirm_token TEXT,
                    firebase_uid TEXT UNIQUE,
                    avatar TEXT,
                    preferred_language TEXT,
                    detection_mode TEXT,
                    camera TEXT,
                    output_type TEXT,
                    hand_mode TEXT,
                    camera_sensitivity INTEGER DEFAULT 700,
                    total_signs INTEGER DEFAULT 0,
                    last_detection TEXT,
                    reset_token TEXT,
                    reset_expiry TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')

            # Create detection_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp TEXT,
                    text TEXT,
                    confidence REAL,
                    raw_label TEXT,
                    FOREIGN KEY (user_id) REFERENCES [user] (id)
                )
            ''')

            # Create contact_messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contact_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    subject TEXT,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create newsletter_subscribers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS newsletter_subscribers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    subscribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create chat_rooms table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_rooms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_name TEXT NOT NULL,
                    description TEXT,
                    room_type TEXT NOT NULL,
                    created_by INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (created_by) REFERENCES [user] (id)
                )
            ''')

            # Create messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    message_text TEXT NOT NULL,
                    message_type TEXT DEFAULT 'text',
                    audio_url TEXT,
                    video_url TEXT,
                    is_sign_detected INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (room_id) REFERENCES chat_rooms (id),
                    FOREIGN KEY (user_id) REFERENCES [user] (id)
                )
            ''')

            # Create private_messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS private_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id INTEGER NOT NULL,
                    receiver_id INTEGER NOT NULL,
                    message_text TEXT NOT NULL,
                    is_read INTEGER DEFAULT 0,
                    is_sign_detected INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sender_id) REFERENCES [user] (id),
                    FOREIGN KEY (receiver_id) REFERENCES [user] (id)
                )
            ''')

            # Create online_users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS online_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    current_room_id INTEGER,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES [user] (id),
                    FOREIGN KEY (current_room_id) REFERENCES chat_rooms (id)
                )
            ''')

            # Create room_members table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS room_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    room_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (room_id) REFERENCES chat_rooms (id),
                    FOREIGN KEY (user_id) REFERENCES [user] (id),
                    UNIQUE(room_id, user_id)
                )
            ''')

        conn.commit()
        conn.close()
        
        migrate_messages_table(database_path)
    except Exception as e:
        print(f"Error initializing database ({database_path}): {e}")

def migrate_messages_table(database_path):
    try:
        if "DRIVER=" in database_path:
            import pyodbc
            conn = pyodbc.connect(database_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    IF NOT EXISTS (SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME='messages' AND COLUMN_NAME='message_type')
                    ALTER TABLE messages ADD message_type NVARCHAR(50) DEFAULT 'text'
                ''')
                conn.commit()
                print("Added message_type column to messages table")
            except Exception as e:
                print(f"message_type column already exists or error: {e}")
            
            try:
                cursor.execute('''
                    IF NOT EXISTS (SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME='messages' AND COLUMN_NAME='audio_url')
                    ALTER TABLE messages ADD audio_url NVARCHAR(MAX)
                ''')
                conn.commit()
                print("Added audio_url column to messages table")
            except Exception as e:
                print(f"audio_url column already exists or error: {e}")
            
            try:
                cursor.execute('''
                    IF NOT EXISTS (SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME='messages' AND COLUMN_NAME='video_url')
                    ALTER TABLE messages ADD video_url NVARCHAR(MAX)
                ''')
                conn.commit()
                print("Added video_url column to messages table")
            except Exception as e:
                print(f"video_url column already exists or error: {e}")
            
            conn.close()
        else:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            
            cursor.execute("PRAGMA table_info(messages)")
            columns = {row[1] for row in cursor.fetchall()}
            
            if 'message_type' not in columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN message_type TEXT DEFAULT "text"')
                conn.commit()
                print("Added message_type column to messages table (SQLite)")
            
            if 'audio_url' not in columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN audio_url TEXT')
                conn.commit()
                print("Added audio_url column to messages table (SQLite)")
            
            if 'video_url' not in columns:
                cursor.execute('ALTER TABLE messages ADD COLUMN video_url TEXT')
                conn.commit()
                print("Added video_url column to messages table (SQLite)")
            
            conn.close()
    except Exception as e:
        print(f"Error migrating messages table: {e}")
