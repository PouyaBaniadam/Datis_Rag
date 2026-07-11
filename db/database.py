import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "Data")

os.makedirs(DB_DIR, exist_ok=True)

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