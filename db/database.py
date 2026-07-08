import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Dynamically locate the project root directory (Datis/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Target the Data folder absolutely
DB_DIR = os.path.join(BASE_DIR, "Data")

# Ensure the Data directory exists
os.makedirs(DB_DIR, exist_ok=True)

# Define the absolute path for datis.db inside Data/
DATABASE_PATH = os.path.join(DB_DIR, "datis.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

print(f"[Database] Target database path: {DATABASE_PATH}")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()