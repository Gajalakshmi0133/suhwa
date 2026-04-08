from backend.utils.db import init_db
from config import Config

init_db(Config.DATABASE)
print("Database initialized successfully!")
