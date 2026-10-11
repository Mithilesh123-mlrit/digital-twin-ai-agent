import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import agent

app = FastAPI(title="Digital Twin")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this if you build a separate frontend
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str


def extract_text(content) -> str:
    """Gemini sometimes returns a list of parts instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content)


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q.question}]}
        )
        return {"answer": extract_text(result["messages"][-1].content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>My Digital Twin</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 16px; }
    #chat { border: 1px solid #ccc; border-radius: 8px; padding: 12px; height: 400px; overflow-y: auto; }
    .you { color: #0a58ca; margin: 8px 0; }
    .bot { margin: 8px 0; white-space: pre-wrap; }
    #row { display: flex; gap: 8px; margin-top: 12px; }
    input { flex: 1; padding: 10px; }
    button { padding: 10px 16px; }
  </style>
</head>
<body>
  <h2>My Digital Twin</h2>
  <p>Ask about my skills, projects, experience, or have me calculate or search.</p>
  <div id="chat"></div>
  <div id="row">
    <input id="q" placeholder="Type your question..." />
    <button onclick="send()">Send</button>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const input = document.getElementById("q");
    input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });

    function add(cls, text) {
      const d = document.createElement("div");
      d.className = cls;
      d.textContent = text;
      chat.appendChild(d);
      chat.scrollTop = chat.scrollHeight;
      return d;
    }

    async function send() {
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      add("you", "You: " + text);
      const wait = add("bot", "Digital Twin: thinking...");
      try {
        const r = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text })
        });
        const data = await r.json();
        wait.textContent = "Digital Twin: " + (data.answer || data.detail);
      } catch (e) {
        wait.textContent = "Digital Twin: Sorry, something went wrong.";
      }
    }
  </script>
</body>
</html>
"""
