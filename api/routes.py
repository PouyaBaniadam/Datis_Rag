import json
import os
import shutil
import traceback
import asyncio
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, WebSocket, WebSocketDisconnect, Form
from sqlalchemy.orm import Session
from typing import List

from api.schemas import ChatRequest, SessionCreate, SessionResponse, MessageResponse
from core.llm_manager import LLMManager
from core.retriever import Retriever
from db.database import get_db
from db.models import ChatSession, ChatMessage

router = APIRouter()
llm_manager = LLMManager()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")

os.makedirs(DATA_DIR, exist_ok=True)

retriever = None


def get_retriever():
    global retriever
    if retriever is not None:
        return retriever
    if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH):
        try:
            retriever = Retriever(
                chunks_path=CHUNKS_PATH,
                metadata_path=METADATA_PATH,
                embeddings_path=EMBEDDINGS_PATH,
                device="cpu"
            )
            return retriever
        except Exception as e:
            return None
    return None


@router.get("/status")
def get_system_status():
    files_exist = os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH)
    pdf_files = []
    if os.path.exists(DATA_DIR):
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    return {
        "is_indexed": files_exist,
        "uploaded_files": pdf_files,
        "message": "Ready to chat" if files_exist else "No documents uploaded yet"
    }


@router.post("/sessions", response_model=SessionResponse)
def create_session(session_data: SessionCreate, db: Session = Depends(get_db)):
    db_session = ChatSession(title=session_data.title)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


@router.get("/sessions", response_model=List[SessionResponse])
def get_sessions(db: Session = Depends(get_db)):
    return db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()


@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
def get_session_messages(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(
        ChatMessage.created_at.asc()
    ).all()


@router.post("/upload")
def upload_pdf_file(file: UploadFile = File(...), embedding_type: str = Form("text-embedding-3-small")):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        file_location = os.path.join(DATA_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from core.build_embeddings import EmbeddingBuilder
        builder = EmbeddingBuilder(embedding_type=embedding_type)
        builder.build()

        global retriever
        retriever = None

        return {"message": "File processed and indexed successfully."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@router.post("/chat")
def chat_with_document(request: ChatRequest, db: Session = Depends(get_db)):
    current_retriever = get_retriever()
    if not current_retriever:
        raise HTTPException(
            status_code=400,
            detail="System data not found. Please process documents first."
        )

    db_session = db.query(ChatSession).filter(ChatSession.id == request.session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    try:
        user_msg = ChatMessage(
            session_id=request.session_id,
            role="user",
            content=request.query
        )
        db.add(user_msg)
        db.commit()

        results = current_retriever.retrieve(
            request.query,
            embedding_type=request.embedding_type,
            top_k=request.top_k
        )

        if not results:
            assistant_response = "I could not find relevant information in the documents."
            sources_json = "[]"
        else:
            context_blocks = []
            for i, r in enumerate(results, 1):
                context_blocks.append(f"[{i}] {r['text']}")
            context_text = "\n\n".join(context_blocks)

            prompt = f"Use the following texts to answer the question.\n\nTexts:\n{context_text}\n\nQuestion:\n{request.query}"

            assistant_response = llm_manager.generate(prompt, request.model_type)

            simplified_sources = [
                {
                    "text": r["text"],
                    "score": r["score"],
                    "doc_id": r["metadata"]["doc_id"],
                    "page_number": r["metadata"].get("page_number")
                } for r in results
            ]
            sources_json = json.dumps(simplified_sources, ensure_ascii=False)

        assistant_msg = ChatMessage(
            session_id=request.session_id,
            role="assistant",
            content=assistant_response,
            sources=sources_json
        )
        db.add(assistant_msg)
        db.commit()

        return {
            "answer": assistant_response,
            "chunks": results,
            "session_id": request.session_id
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    print("\n[WS Server] Connecting request received...")
    await websocket.accept()
    print("[WS Server] Connection accepted and established successfully.")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"[WS Server] Raw payload received from Flutter: {data}")
            request_data = json.loads(data)

            query = request_data["query"]
            session_id = request_data["session_id"]
            model_type = request_data["model_type"]
            embedding_type = request_data.get("embedding_type", "api")
            top_k = request_data.get("top_k", 3)

            print(f"[WS Server] Saving User message in database for Session ID: {session_id}")
            user_msg = ChatMessage(session_id=session_id, role="user", content=query)
            db.add(user_msg)
            db.commit()

            print("[WS Server] Initializing/fetching retriever...")
            current_retriever = get_retriever()
            if not current_retriever:
                print("[WS Server] Error: Retriever could not be loaded!")
                await websocket.send_json({"type": "error", "content": "System data not found."})
                continue

            print(f"[WS Server] Searching document context chunks via Retriever (Embedding: {embedding_type})")
            results = current_retriever.retrieve(query, embedding_type=embedding_type, top_k=top_k)
            print(f"[WS Server] Retrieval complete. Found {len(results)} context chunks.")

            simplified_sources = [
                {
                    "text": r["text"],
                    "score": r["score"],
                    "doc_id": r["metadata"]["doc_id"],
                    "page_number": r["metadata"].get("page_number")
                } for r in results
            ]

            print("[WS Server] Sending source references to Flutter Client...")
            await websocket.send_json({"type": "sources", "content": simplified_sources})

            context_blocks = [f"[{i}] {r['text']}" for i, r in enumerate(results, 1)]
            context_text = "\n\n".join(context_blocks)
            prompt = f"Use the following texts to answer the question.\n\nTexts:\n{context_text}\n\nQuestion:\n{query}"

            print(f"[WS Server] Initiating LLM Generation Stream (Model: {model_type})...")
            full_response = ""

            for token in llm_manager.generate_stream(prompt, model_type):
                full_response += token
                print(f"[WS Server] Yielding token: {repr(token)}")
                await websocket.send_json({"type": "token", "content": token})
                await asyncio.sleep(0.001)

            print(f"[WS Server] LLM Stream finished. Response length: {len(full_response)} chars.")
            print("[WS Server] Saving Assistant full response in database...")

            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_response,
                sources=json.dumps(simplified_sources, ensure_ascii=False)
            )
            db.add(assistant_msg)
            db.commit()

            print("[WS Server] Sending done status to Flutter Client.")
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print("[WS Server] Connection disconnected by client.")
    except Exception as e:
        print(f"[WS Server] Fatal Exception occurred: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass