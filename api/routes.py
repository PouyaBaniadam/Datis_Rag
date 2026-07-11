import json
import os
import shutil
import asyncio
from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    Form,
    Header,
)
from sqlalchemy.orm import Session
from typing import List

from api.schemas import (
    ChatRequest,
    SessionCreate,
    SessionResponse,
    MessageResponse,
)
from core.llm_manager import LLMManager
from core.retriever import Retriever
from db.database import get_db
from db.models import (
    ChatSession,
    ChatMessage,
)

router = APIRouter()
llm_manager = LLMManager()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")

os.makedirs(DATA_DIR, exist_ok=True)

retrievers = {}


def get_retriever(session_id: int):
    if session_id in retrievers:
        return retrievers[session_id]

    chunks_path = os.path.join(DATA_DIR, f"session_{session_id}_chunks.json")
    metadata_path = os.path.join(DATA_DIR, f"session_{session_id}_metadata.json")
    embeddings_path = os.path.join(DATA_DIR, f"session_{session_id}_embeddings.npy")

    if os.path.exists(chunks_path) and os.path.exists(embeddings_path):
        try:
            r = Retriever(
                chunks_path=chunks_path,
                metadata_path=metadata_path,
                embeddings_path=embeddings_path,
                device="cpu"
            )
            retrievers[session_id] = r
            return r
        except Exception:
            return None
    return None


def extract_api_key(authorization: str) -> str:
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ")[1]
    return None


@router.get("/status")
def get_system_status():
    pdf_files = []
    if os.path.exists(DATA_DIR):
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    return {
        "uploaded_files": pdf_files,
        "message": "سیستم آماده کار است"
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
        raise HTTPException(status_code=404, detail="جلسه گفتگو پیدا نشد")
    return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(
        ChatMessage.created_at.asc()
    ).all()


@router.post("/upload")
def upload_pdf_file(
        file: UploadFile = File(...),
        embedding_type: str = Form("text-embedding-3-small"),
        session_id: int = Form(...),
        authorization: str = Header(None)
):
    try:
        api_key = extract_api_key(authorization)
        if embedding_type != "local" and not api_key:
            raise HTTPException(status_code=401,
                                detail="کلید API یافت نشد. لطفاً ابتدا از طریق آیکون کلید در هدر صفحه، کلید اختصاصی خود را ثبت کنید.")
        os.makedirs(DATA_DIR, exist_ok=True)
        file_location = os.path.join(DATA_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        from core.build_embeddings import EmbeddingBuilder
        builder = EmbeddingBuilder(session_id=session_id, embedding_type=embedding_type, api_key=api_key)
        builder.build()
        if session_id in retrievers:
            del retrievers[session_id]
        return {"message": "سند با موفقیت پردازش و ذخیره شد."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"پردازش سند ناموفق بود: {str(e)}")


@router.post("/chat")
def chat_with_document(
        request: ChatRequest,
        db: Session = Depends(get_db),
        authorization: str = Header(None)
):
    current_retriever = get_retriever(request.session_id)
    if not current_retriever:
        raise HTTPException(
            status_code=400,
            detail="داده‌های سیستم برای این گفتگو یافت نشد. لطفاً ابتدا سند خود را آپلود کنید."
        )
    db_session = db.query(ChatSession).filter(ChatSession.id == request.session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="جلسه گفتگو پیدا نشد")
    try:
        api_key = extract_api_key(authorization)
        is_online_model = "gemma" not in request.model_type.lower() and "mistral" not in request.model_type.lower()
        is_online_emb = request.embedding_type != "local"
        if (is_online_model or is_online_emb) and not api_key:
            raise HTTPException(status_code=401,
                                detail="کلید API یافت نشد. لطفاً ابتدا از طریق آیکون کلید در هدر صفحه، کلید اختصاصی خود را ثبت کنید.")
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
            top_k=request.top_k,
            api_key=api_key
        )
        if not results:
            assistant_response = "اطلاعات مرتبطی در اسناد پیدا نشد."
            sources_json = "[]"
        else:
            context_blocks = []
            for i, r in enumerate(results, 1):
                context_blocks.append(f"[{i}] {r['text']}")
            context_text = "\n\n".join(context_blocks)
            prompt = f"با استفاده از متن‌های زیر، به سوال پاسخ دهید. حتماً پاسخ خود را به زبان فارسی بنویسید.\n\nمتن‌ها:\n{context_text}\n\nسوال:\n{request.query}"
            assistant_response = llm_manager.generate(prompt, request.model_type, api_key=api_key)
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
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            query = request_data["query"]
            session_id = request_data["session_id"]
            model_type = request_data["model_type"]
            embedding_type = request_data.get("embedding_type", "api")
            top_k = request_data.get("top_k", 3)
            api_key = request_data.get("api_key")
            is_online_model = "gemma" not in model_type.lower() and "mistral" not in model_type.lower()
            is_online_emb = embedding_type != "local"
            if (is_online_model or is_online_emb) and not api_key:
                await websocket.send_json({"type": "error",
                                           "content": "کلید API یافت نشد. لطفاً ابتدا از طریق آیکون کلید در هدر صفحه، کلید اختصاصی خود را ثبت کنید."})
                continue
            user_msg = ChatMessage(session_id=session_id, role="user", content=query)
            db.add(user_msg)
            db.commit()
            current_retriever = get_retriever(session_id)
            if not current_retriever:
                await websocket.send_json({"type": "error",
                                           "content": "داده‌های سیستم برای این گفتگو یافت نشد. لطفاً ابتدا سند خود را آپلود کنید."})
                continue
            results = current_retriever.retrieve(query, embedding_type=embedding_type, top_k=top_k, api_key=api_key)
            simplified_sources = [
                {
                    "text": r["text"],
                    "score": r["score"],
                    "doc_id": r["metadata"]["doc_id"],
                    "page_number": r["metadata"].get("page_number")
                } for r in results
            ]
            await websocket.send_json({"type": "sources", "content": simplified_sources})
            context_blocks = [f"[{i}] {r['text']}" for i, r in enumerate(results, 1)]
            context_text = "\n\n".join(context_blocks)
            prompt = f"با استفاده از متن‌های زیر، به سوال پاسخ دهید. حتماً پاسخ خود را به زبان فارسی بنویسید.\n\nمتن‌ها:\n{context_text}\n\nسوال:\n{query}"
            full_response = ""
            for token in llm_manager.generate_stream(prompt, model_type, api_key=api_key):
                full_response += token
                await websocket.send_json({"type": "token", "content": token})
                await asyncio.sleep(0.001)
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_response,
                sources=json.dumps(simplified_sources, ensure_ascii=False)
            )
            db.add(assistant_msg)
            db.commit()
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


@router.put("/sessions/{session_id}", response_model=SessionResponse)
def rename_session(session_id: int, session_data: SessionCreate, db: Session = Depends(get_db)):
    db_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="جلسه گفتگو پیدا نشد")
    db_session.title = session_data.title
    db.commit()
    db.refresh(db_session)
    return db_session


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    db_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="جلسه گفتگو پیدا نشد")

    db.delete(db_session)
    db.commit()

    chunks_path = os.path.join(DATA_DIR, f"session_{session_id}_chunks.json")
    metadata_path = os.path.join(DATA_DIR, f"session_{session_id}_metadata.json")
    embeddings_path = os.path.join(DATA_DIR, f"session_{session_id}_embeddings.npy")

    if os.path.exists(chunks_path):
        os.remove(chunks_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    if os.path.exists(embeddings_path):
        os.remove(embeddings_path)

    if session_id in retrievers:
        del retrievers[session_id]

    return {"message": "گفتگو با موفقیت حذف شد"}