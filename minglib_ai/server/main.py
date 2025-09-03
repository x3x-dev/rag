from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
from datetime import datetime
import time
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

# Import our models
from models.simple_rag import get_simple_response
from models.no_rag import get_response as get_no_rag_response
from models.enhanced_rag import init_enhanced_rag, get_enhanced_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # hybrid by default; set enable_rerank=True for tougher queries (slower)
    init_enhanced_rag(True)
    yield
    # Cleanup code here

app = FastAPI(
    title="MingLib AI Assistant API",
    description="RAG-powered assistant for MingLib quantitative finance library",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for all chat messages
chat_history: List[Dict] = []

# Request/Response models
class ChatRequest(BaseModel):
    question: str
    model: str = "simple"  # Options: simple, enhanced, no_rag

class ChatResponse(BaseModel):
    success: bool
    data: Optional[Dict] = None
    error: Optional[Dict] = None

class MessageData(BaseModel):
    id: str
    content: str
    reference: str
    model_used: str
    timestamp: str
    processing_time_ms: int




@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "MingLib AI Assistant API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint - supports both simple and enhanced models"""
    try:
        # Validate model parameter
        if request.model not in ["simple", "enhanced", "no_rag"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error": {
                        "message": "Invalid model. Must be 'simple', 'enhanced', or 'no_rag'",
                        "code": "INVALID_MODEL",
                        "details": {"provided_model": request.model}
                    }
                }
            )
        
        # Get response from appropriate model
        if request.model == "no_rag":
            response_data = get_no_rag_response(request.question)
        elif request.model == "enhanced":
            response_data = get_enhanced_response(request.question)
        else: 
            response_data = get_simple_response(request.question)
        
        # Add user message to history
        user_message = {
            "id": f"msg_{int(time.time() * 1000)}",
            "content": request.question,
            "is_user": True,
            "model_requested": request.model,
            "timestamp": datetime.now().isoformat()
        }
        chat_history.append(user_message)
        
        # Add AI response to history
        ai_message = {
            "id": response_data["data"]["id"],
            "content": response_data["data"]["content"],
            "reference": response_data["data"]["reference"],
            "model_used": request.model,
            "is_user": False,
            "timestamp": response_data["data"]["timestamp"],
            "processing_time_ms": response_data["data"]["processing_time_ms"]
        }
        chat_history.append(ai_message)
        
        return ChatResponse(**response_data)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": {
                    "message": str(e),
                    "code": "CHAT_ERROR",
                    "details": {"question": request.question, "model": request.model}
                }
            }
        )

@app.get("/api/chat/history")
async def get_chat_history():
    """Get all chat history"""
    return {
        "messages": chat_history,
        "message_count": len(chat_history)
    }

@app.delete("/api/chat/history")
async def clear_chat_history():
    """Clear all chat history"""
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
