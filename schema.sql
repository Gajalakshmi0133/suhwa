-- SQL Schema for Suhwa Application (SQL Server / T-SQL)

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='users' AND xtype='U')
CREATE TABLE users (
    id INT PRIMARY KEY IDENTITY(1,1),
    name NVARCHAR(255) NOT NULL,
    email NVARCHAR(255) UNIQUE NOT NULL,
    username NVARCHAR(255) UNIQUE NOT NULL,
    password_hash NVARCHAR(MAX) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

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
    xp_points INT DEFAULT 0,
    daily_streak INT DEFAULT 0,
    last_practice_date DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='detection_history' AND xtype='U')
CREATE TABLE detection_history (
    id INT PRIMARY KEY IDENTITY(1,1),
    user_id INT,
    timestamp NVARCHAR(100),
    text NVARCHAR(MAX),
    confidence FLOAT,
    raw_label NVARCHAR(255),
    FOREIGN KEY (user_id) REFERENCES [user] (id)
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user_progress' AND xtype='U')
CREATE TABLE user_progress (
    id INT PRIMARY KEY IDENTITY(1,1),
    user_id INT,
    lesson_id NVARCHAR(100),
    status NVARCHAR(50) DEFAULT 'pending', -- pending, in_progress, completed
    accuracy FLOAT DEFAULT 0.0,
    attempts INT DEFAULT 0,
    last_practice DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES [user] (id)
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='chat_rooms' AND xtype='U')
CREATE TABLE chat_rooms (
    id INT PRIMARY KEY IDENTITY(1,1),
    room_name NVARCHAR(255) NOT NULL,
    description NVARCHAR(MAX),
    room_type NVARCHAR(50) NOT NULL, -- 'public', 'topic'
    created_by INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES [user] (id)
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='messages' AND xtype='U')
CREATE TABLE messages (
    id INT PRIMARY KEY IDENTITY(1,1),
    room_id INT NOT NULL,
    user_id INT NOT NULL,
    message_text NVARCHAR(MAX) NOT NULL,
    message_type NVARCHAR(50) DEFAULT 'text', -- 'text', 'audio', 'video'
    audio_url NVARCHAR(MAX),
    video_url NVARCHAR(MAX),
    is_sign_detected INT DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (room_id) REFERENCES chat_rooms (id),
    FOREIGN KEY (user_id) REFERENCES [user] (id)
);

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
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='online_users' AND xtype='U')
CREATE TABLE online_users (
    id INT PRIMARY KEY IDENTITY(1,1),
    user_id INT NOT NULL,
    current_room_id INT,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES [user] (id),
    FOREIGN KEY (current_room_id) REFERENCES chat_rooms (id)
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='room_members' AND xtype='U')
CREATE TABLE room_members (
    id INT PRIMARY KEY IDENTITY(1,1),
    room_id INT NOT NULL,
    user_id INT NOT NULL,
    joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (room_id) REFERENCES chat_rooms (id),
    FOREIGN KEY (user_id) REFERENCES [user] (id),
    UNIQUE(room_id, user_id)
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user_encryption_keys' AND xtype='U')
CREATE TABLE user_encryption_keys (
    id INT PRIMARY KEY IDENTITY(1,1),
    user_id INT NOT NULL,
    public_key NVARCHAR(MAX) NOT NULL,
    private_key NVARCHAR(MAX) NOT NULL,
    key_algorithm NVARCHAR(50) DEFAULT 'RSA-2048',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES [user] (id),
    UNIQUE(user_id)
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='messages' AND xtype='U')
BEGIN
    ALTER TABLE messages ADD 
        is_encrypted INT DEFAULT 0,
        encryption_algorithm NVARCHAR(50),
        message_signature NVARCHAR(MAX);
END

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='private_messages' AND xtype='U')
BEGIN
    ALTER TABLE private_messages ADD 
        is_encrypted INT DEFAULT 0,
        encryption_algorithm NVARCHAR(50),
        message_signature NVARCHAR(MAX);
END
