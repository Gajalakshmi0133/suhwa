import sqlite3
import os

db_path = 'instance/app.db'
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    # try config
    from config import Config
    db_path = Config.DATABASE
    print(f"Using DB from config: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    print("Checking 'users' table for gaj13:")
    c.execute("SELECT id, username FROM users WHERE username='gaj13' OR username='Gaja'")
    print(c.fetchall())
    
    print("\nChecking '[user]' table for gaj13:")
    c.execute("SELECT id, username FROM [user] WHERE username='gaj13' OR username='Gaja'")
    print(c.fetchall())
    
    print("\nRecent messages in room 7:")
    c.execute("SELECT id, room_id, user_id, message_text FROM messages WHERE room_id=7 ORDER BY timestamp DESC LIMIT 5")
    print(c.fetchall())
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
