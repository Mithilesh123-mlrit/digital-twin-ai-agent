from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import agent
from retriever import retriever


# ==========================================================
# FastAPI application
# ==========================================================

app = FastAPI(
    title="Digital Twin AI API",
    version="1.0.0",
    description=(
        "Personal Digital Twin using Gemini, LangChain, "
        "RAG, Chroma, Calculator and Web Search."
    )
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# In-memory conversation sessions
# ==========================================================

sessions: dict[str, list[Any]] = {}


# ==========================================================
# Request models
# ==========================================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = "default"


class ResumeSearchRequest(BaseModel):
    query: str = Field(min_length=1)


# ==========================================================
# Helper
# ==========================================================

def extract_text(content) -> str:
    """
    Convert Gemini/LangChain message content into plain text.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:

            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    text_parts.append(text)

            else:
                text_parts.append(str(item))

        return "\n".join(text_parts)

    return str(content)


# ==========================================================
# GET - Root
# ==========================================================

@app.get("/")
def root():

    return {
        "message": "Welcome to the Digital Twin AI API",
        "features": [
            "Resume RAG",
            "Gemini LLM",
            "Calculator",
            "Web Search",
            "Conversation Memory"
        ]
    }


# ==========================================================
# GET - Health Check
# ==========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==========================================================
# GET - Available Tools
# ==========================================================

@app.get("/tools")
def get_tools():

    return {
        "tools": [
            {
                "name": "get_resume_data",
                "description": "Search personal resume using RAG"
            },
            {
                "name": "calculator",
                "description": "Perform mathematical calculations"
            },
            {
                "name": "web_search",
                "description": "Search current information on the web"
            }
        ]
    }


# ==========================================================
# POST - Direct Resume RAG Search
# ==========================================================

@app.post("/resume/search")
def resume_search(request: ResumeSearchRequest):

    try:

        documents = retriever.invoke(request.query)

        results = []

        for document in documents:

            results.append(
                {
                    "content": document.page_content,
                    "metadata": document.metadata
                }
            )

        return {
            "query": request.query,
            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# POST - Talk to Digital Twin
# ==========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        messages = sessions.setdefault(
            request.session_id,
            []
        )

        messages.append(
            {
                "role": "user",
                "content": request.message
            }
        )

        result = agent.invoke(
            {
                "messages": messages
            }
        )

        # Store updated conversation
        sessions[request.session_id] = result["messages"]

        final_message = result["messages"][-1]

        return {
            "session_id": request.session_id,
            "response": extract_text(final_message.content)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================================
# GET - Session Information
# ==========================================================

@app.get("/sessions/{session_id}")
def get_session(session_id: str):

    if session_id not in sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    return {
        "session_id": session_id,
        "message_count": len(sessions[session_id])
    }


# ==========================================================
# PUT - Create / Reset Session
# ==========================================================

@app.put("/sessions/{session_id}")
def reset_session(session_id: str):

    sessions[session_id] = []

    return {
        "message": "Session created/reset successfully",
        "session_id": session_id
    }


# ==========================================================
# DELETE - Delete Conversation Session
# ==========================================================

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):

    if session_id not in sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    del sessions[session_id]

    return {
        "message": f"Session '{session_id}' deleted successfully"
    }