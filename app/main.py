"""
FastAPI application for the Context-Managed Multi-Turn Chatbot.
Provides REST endpoints and serves the web interface.
"""

import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.persona import DEFAULT_BOT_NAME, SYSTEM_PROMPT
from app.context_manager import context_manager
from app.llm_client import llm_client, LLMClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Context-Managed Chatbot API",
    description="Multi-turn LLM Chatbot with Context Window Management & Persona",
    version="1.0.0",
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Pydantic Request / Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user message to send to the bot.")
    conversation_id: Optional[str] = Field(None, description="Unique conversation session ID.")
    strategy: Optional[str] = Field(None, description="Context window strategy: 'sliding_window' or 'summarization'.")
    max_messages: Optional[int] = Field(None, ge=2, le=50, description="Max messages in direct context window.")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt override.")


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    stats: Dict[str, Any]
    model: str
    bot_name: str = DEFAULT_BOT_NAME


@app.exception_handler(LLMClientError)
async def llm_client_error_handler(request: Request, exc: LLMClientError):
    """Custom exception handler for LLM errors with clear friendly messages."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.message,
            "details": exc.details,
            "status_code": exc.status_code
        }
    )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the web chat interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Context-Managed Chatbot API is running.</h1><p>Static UI not found.</p>")


@app.get("/api/health")
async def health_check():
    """Returns application health, API key status, and configuration details."""
    api_key_set = settings.is_api_key_configured()
    return {
        "status": "healthy",
        "api_key_configured": api_key_set,
        "model": settings.GEMINI_MODEL,
        "default_strategy": settings.CONTEXT_STRATEGY,
        "max_context_messages": settings.MAX_CONTEXT_MESSAGES,
        "active_conversations_count": len(context_manager.list_all()),
        "bot_persona": DEFAULT_BOT_NAME
    }


@app.post("/api/config/reload")
async def reload_config():
    """Reloads settings from .env file without restarting the server."""
    settings.reload()
    return {
        "reloaded": True,
        "api_key_configured": settings.is_api_key_configured(),
        "model": settings.GEMINI_MODEL,
        "strategy": settings.CONTEXT_STRATEGY,
        "max_messages": settings.MAX_CONTEXT_MESSAGES
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint:
    - Retrieves or creates conversation session
    - Applies context window management (sliding window or progressive summarization)
    - Queries Gemini model with system persona
    - Returns model response and context statistics
    """
    # Generate or validate conversation_id
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Retrieve or initialize the conversation instance
    conversation = context_manager.get_or_create(
        conversation_id=conversation_id,
        strategy=request.strategy,
        max_messages=request.max_messages
    )

    # Generate response
    reply_text, stats = llm_client.generate_response(
        conversation=conversation,
        user_message=request.message,
        custom_system_prompt=request.system_prompt
    )

    return ChatResponse(
        response=reply_text,
        conversation_id=conversation_id,
        stats=stats,
        model=settings.GEMINI_MODEL,
        bot_name=DEFAULT_BOT_NAME
    )


@app.get("/api/conversations")
async def list_conversations():
    """List all active conversation sessions and their context window stats."""
    return {
        "conversations": context_manager.list_all()
    }


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get full message history and context details for a conversation."""
    conversation = context_manager.get(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found."
        )

    return {
        "conversation_id": conversation.conversation_id,
        "stats": conversation.get_stats(),
        "messages": [m.to_dict() for m in conversation.messages],
        "running_summary": conversation.running_summary
    }


@app.delete("/api/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Reset the history and summary of a specific conversation."""
    conversation = context_manager.get(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found."
        )

    conversation.clear()
    return {
        "cleared": True,
        "conversation_id": conversation_id,
        "stats": conversation.get_stats()
    }


@app.post("/api/conversations/new")
async def create_new_conversation():
    """Generate a new conversation session ID."""
    new_id = str(uuid.uuid4())
    conversation = context_manager.get_or_create(new_id)
    return {
        "conversation_id": new_id,
        "stats": conversation.get_stats()
    }
