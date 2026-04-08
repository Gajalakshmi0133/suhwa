import sys
from werkzeug.security import generate_password_hash
from backend.utils.db import get_db_connection
from config import Config
from datetime import datetime

def migrate_sample_users():
    """Add 10 sample users to the database"""
    conn = get_db_connection(Config.DATABASE)
    cursor = conn.cursor()
    
    sample_users = [
        {
            'username': 'alice_sign',
            'email': 'alice@suhwa.com',
            'first_name': 'Alice',
            'last_name': 'Johnson',
            'password': 'Password123'
        },
        {
            'username': 'bob_practice',
            'email': 'bob@suhwa.com',
            'first_name': 'Bob',
            'last_name': 'Smith',
            'password': 'Password123'
        },
        {
            'username': 'carol_learner',
            'email': 'carol@suhwa.com',
            'first_name': 'Carol',
            'last_name': 'Williams',
            'password': 'Password123'
        },
        {
            'username': 'david_tutor',
            'email': 'david@suhwa.com',
            'first_name': 'David',
            'last_name': 'Brown',
            'password': 'Password123'
        },
        {
            'username': 'emma_helper',
            'email': 'emma@suhwa.com',
            'first_name': 'Emma',
            'last_name': 'Davis',
            'password': 'Password123'
        },
        {
            'username': 'frank_mentor',
            'email': 'frank@suhwa.com',
            'first_name': 'Frank',
            'last_name': 'Miller',
            'password': 'Password123'
        },
        {
            'username': 'grace_expert',
            'email': 'grace@suhwa.com',
            'first_name': 'Grace',
            'last_name': 'Wilson',
            'password': 'Password123'
        },
        {
            'username': 'henry_student',
            'email': 'henry@suhwa.com',
            'first_name': 'Henry',
            'last_name': 'Moore',
            'password': 'Password123'
        },
        {
            'username': 'iris_teacher',
            'email': 'iris@suhwa.com',
            'first_name': 'Iris',
            'last_name': 'Taylor',
            'password': 'Password123'
        },
        {
            'username': 'jack_enthusiast',
            'email': 'jack@suhwa.com',
            'first_name': 'Jack',
            'last_name': 'Anderson',
            'password': 'Password123'
        }
    ]
    
    try:
        for user in sample_users:
            cursor.execute('SELECT COUNT(*) FROM [user] WHERE username = ?', (user['username'],))
            if cursor.fetchone()[0] == 0:
                cursor.execute('''INSERT INTO [user] (username, email, first_name, last_name, password, is_confirmed, created_at)
                                VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)''',
                              (user['username'], user['email'], user['first_name'], 
                               user['last_name'], generate_password_hash(user['password'])))
        
        conn.commit()
        print("[OK] 10 sample users added successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error adding users: {e}")
        conn.rollback()
    finally:
        conn.close()

def migrate_default_rooms():
    """Create 5 default room types"""
    conn = get_db_connection(Config.DATABASE)
    cursor = conn.cursor()
    
    default_rooms = [
        {
            'room_name': 'General Discussion',
            'description': 'Welcome to the general discussion room. Practice sign conversations and ask questions!',
            'room_type': 'public'
        },
        {
            'room_name': 'Alphabet Practice',
            'description': 'Practice and master the sign language alphabet with fellow learners.',
            'room_type': 'topic'
        },
        {
            'room_name': 'Numbers & Counting',
            'description': 'Learn and practice sign language numbers and counting techniques.',
            'room_type': 'topic'
        },
        {
            'room_name': 'Daily Words',
            'description': 'Discover and discuss common everyday words and phrases in sign language.',
            'room_type': 'topic'
        },
        {
            'room_name': 'Doubt Discussion',
            'description': 'Ask questions, clear doubts, and get help from experienced signers.',
            'room_type': 'topic'
        }
    ]
    
    try:
        for room in default_rooms:
            cursor.execute('SELECT COUNT(*) FROM chat_rooms WHERE room_name = ?', (room['room_name'],))
            if cursor.fetchone()[0] == 0:
                cursor.execute('''INSERT INTO chat_rooms (room_name, description, room_type, created_by, created_at)
                                VALUES (?, ?, ?, NULL, CURRENT_TIMESTAMP)''',
                              (room['room_name'], room['description'], room['room_type']))
        
        conn.commit()
        print("[OK] 5 default rooms created successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error creating rooms: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    print("[START] Starting migration...\n")
    migrate_sample_users()
    migrate_default_rooms()
    print("\n[COMPLETE] Migration complete!")
