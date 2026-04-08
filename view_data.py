import sqlite3
import os

db_path = r"e:\gajalakshmi\project\Suhwa\instance\app.db"

def query_table(table_name):
    print(f"\n--- Data from {table_name} ---")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM [{table_name}] LIMIT 5")
        colnames = [description[0] for description in cursor.description]
        print(" | ".join(colnames))
        rows = cursor.fetchall()
        for row in rows:
            print(" | ".join(str(val) for val in row))
        conn.close()
    except Exception as e:
        print(f"Error querying {table_name}: {e}")

tables = ['users', 'user', 'user_progress', 'detection_history']
for t in tables:
    query_table(t)
