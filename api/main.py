from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.routes import router

import db.models
from db.database import engine, Base

# This will now automatically create the 'datis.db' file and all its tables on startup
print("[Database] Initializing SQLite database schema...")
Base.metadata.create_all(bind=engine)
print("[Database] Schema synchronized successfully.")

app = FastAPI(
    title="Datis RAG API",
    description="API for Persian RAG System (Online & Offline)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Datis RAG API!", "status": "Active"}

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    print("Starting Datis Server on http://localhost:8000")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)