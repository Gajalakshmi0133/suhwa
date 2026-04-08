import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Ensure folders exist
os.makedirs(os.path.join(BASE_DIR, 'instance'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'uploads'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'static', 'models'), exist_ok=True)

class Config:
    SECRET_KEY = 'f4a9b3c8d2e1f5a7b9c6d4e3f8a1b2c3d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0'
    # For SQLite (Legacy)
    DATABASE = os.path.join(BASE_DIR, 'instance', 'app.db')
    
    # For SQL Server
    # SQL_SERVER_CONN = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=your_server;DATABASE=your_db;UID=your_user;PWD=your_password'
    # Use a simpler local connection string for development if possible
    SQL_SERVER_CONN = os.environ.get('SQL_SERVER_CONN', 'DRIVER={SQL Server};SERVER=localhost;DATABASE=SuhwaDB;Trusted_Connection=yes;')

    
    # DATABASE = SQL_SERVER_CONN # Use SQL_SERVER_CONN as the DATABASE URI
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MODEL_PATH = os.path.join(BASE_DIR, 'static', 'models', 'sign_model.h5')
    MODEL_PATH_MNIST = os.path.join(BASE_DIR, 'static', 'models', 'smnist.h5')
    MODEL_PATH_ASL = os.path.join(BASE_DIR, 'static', 'models', 'sign_model.h5')
    MODEL_PATH_YOLO = os.path.join(BASE_DIR, 'static', 'models', 'best.onnx')
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB
    # Email (SMTP) settings for account confirmation (set in env or replace here)
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USERNAME = 'homecon333@gmail.com'
    MAIL_PASSWORD = 'nsov mztj pvab tfpe'
    MAIL_USE_TLS = True
    MAIL_DEFAULT_SENDER = 'homecon333@gmail.com'
    ADMIN_EMAIL = 'yellowversecenter@gmail.com'
    
    # Admin Credentials
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')
    
