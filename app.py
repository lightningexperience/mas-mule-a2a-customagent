# app.py v0.7 - A2A-compatible stateless Groq LLM Custom Agent
# Update: Switched from brittle urllib to robust 'requests' library

import os
import json
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse

# --- CHANGED: Using requests instead of urllib ---
import requests

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger("CustomAgentServer")
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="CustomAgent A2A Groq LLM Server")


# -----------------------------------------------------------------------------
# Helper: Call Groq Chat Completions API (Using 'requests')
# -----------------------------------------------------------------------------
def call_groq_llm(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    # Defaulting to the versatile model if not specified
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        logger.error("GROQ_API_KEY is not set in environment.")
        return "Error: GROQ_API_KEY is not configured on the server."

    url = "https://api.groq.com/openai/v1/chat/completions"
    
    # 'requests' handles JSON serialization automatically with the json= parameter
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful, general-purpose AI agent. "
                    "Return concise, direct answers in plain text. "
                    "Do not add routing labels or meta commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    # We still keep the custom User-Agent to be safe against WAFs
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "CustomAgentServer/1.0 (Heroku; Python; requests)",
    }

    try:
        # --- THE SWITCH TO REQUESTS ---
        # This automatically handles connection pooling and encoding
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        # Raise an error immediately if the HTTP status code is 4xx or 5xx
        response.raise_for_status()
        
        resp_json = response.json()
        logger.info(f"Groq LLM call succeeded using model: {model}")
        
        return (
            resp_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    except requests.exceptions.HTTPError as http_err:
        # Captures 401 Unauthorized, 404 Not Found, 429 Rate Limit, etc.
        logger.error(f"Groq HTTP error: {http_err} - Body: {response.text}")
        return f"Sorry, I had trouble contacting the language model (HTTP Error)."
        
    except requests.exceptions.ConnectionError:
        logger.error("Groq Connection error: Could not reach the API endpoint.")
        return "Sorry, I could not connect to the language model."
        
    except requests.exceptions.Timeout:
        logger.error("Groq Timeout error: The request took too long.")
        return "Sorry, the language model timed out."
        
    except Exception as e:
        logger.error(f"Groq unexpected error: {e}")
        return "Sorry, I had trouble contacting the language model."


# -----------------------------------------------------------------------------
# Helper: Extract plain text from "message" envelope
# -----------------------------------------------------------------------------
def extract_text_from_message(message: Dict[str, Any]) -> str:
    parts: List[Dict[str, Any]] = message.get("parts", [])
    texts: List[str] = []

    for p in parts:
        if "text" in p:
            texts.append(str(p["text"]))
        elif p.get("type") == "text/plain" and "value" in p:
            texts.append(str(p["value"]))

    joined = "\n".join(texts).strip()
    logger.info(f"CustomAgentServer:Received Message: {joined[:80]}...")
    return joined


# -----------------------------------------------------------------------------
# Build A2A-compatible message response
# -----------------------------------------------------------------------------
def build_a2a_message_event(reply_text: str) -> Dict[str, Any]:
    import uuid

    return {
        "kind": "message",
        "role": "agent",
        "messageId": str(uuid.uuid4()),
        "parts": [
            {
                "kind": "text",
                "text": reply_text
            }
        ]
    }


# -----------------------------------------------------------------------------
# JSON-RPC handler
# -----------------------------------------------------------------------------
async def handle_jsonrpc(payload: Dict[str, Any]) -> JSONResponse:
    logger.info(f"CustomAgentServer:Root POST raw payload: {payload}")

    jsonrpc_version = payload.get("jsonrpc")
    method = payload.get("method")
    msg_id = payload.get("id")

    if jsonrpc_version != "2.0":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32600, "message": "Invalid JSON-RPC version"}
        }, status_code=400)

    if method != "message/send":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unsupported method: {method}"}
        }, status_code=400)

    # Extract text
    params = payload.get("params", {})
    message = params.get("message", {})
    user_text = extract_text_from_message(message)

    # LLM call
    reply_text = call_groq_llm(user_text)

    # Build A2A event
    event = build_a2a_message_event(reply_text)

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": event
    })


# -----------------------------------------------------------------------------
# /tasks endpoint (A2A direct mode)
# -----------------------------------------------------------------------------
@app.post("/tasks")
async def tasks_endpoint(task_request: Dict[str, Any] = Body(...)) -> JSONResponse:
    task_id = task_request.get("taskId")
    skill_id = task_request.get("skillId")
    inputs = task_request.get("inputs", [])

    logger.info(f"CustomAgentServer:Processing Task ID: {task_id}, Skill ID: {skill_id}")

    # Extract text
    all_texts: List[str] = []
    for inp in inputs:
        for c in inp.get("content", []):
            if c.get("type") == "text/plain" and "value" in c:
                all_texts.append(str(c["value"]))

    user_text = "\n".join(all_texts).strip()
    reply_text = call_groq_llm(user_text)

    return JSONResponse({
        "taskId": task_id,
        "status": "completed",
        "outputs": [
            {
                "kind": "message",
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": reply_text
                    }
                ]
            }
        ]
    })


# -----------------------------------------------------------------------------
# Root JSON-RPC endpoint
# -----------------------------------------------------------------------------
@app.post("/")
async def root_jsonrpc_endpoint(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    return await handle_jsonrpc(payload)


# -----------------------------------------------------------------------------
# Dedicated /json-rpc endpoint
# -----------------------------------------------------------------------------
@app.post("/json-rpc")
async def jsonrpc_endpoint(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    return await handle_jsonrpc(payload)


# -----------------------------------------------------------------------------
# Agent card endpoint
# -----------------------------------------------------------------------------
@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request) -> JSONResponse:
    logger.info("============================================================")
    logger.info("CustomAgentServer:AGENT CARD REQUEST")
    logger.info("============================================================")

    base_url = str(request.base_url).rstrip("/")
    jsonrpc_url = f"{base_url}/json-rpc"

    card = {
        "protocolVersion": "0.3.0",
        "name": "Custom Agent A2A",
        "description": "General purpose LLM queries - Groq-powered custom agent.",
        "url": base_url,
        "preferredTransport": "JSONRPC",
        "capabilities": {
            "pushNotifications": False,
            "streaming": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "securitySchemes": {},
        "version": "1.0.0",
        "skills": [
            {
                "id": "general-llm-query",
                "name": "General LLM Query",
                "description": "Answers general knowledge and LLM questions.",
                "examples": ["hey", "whats the capital of india", "Tell me a fun fact"],
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
                "tags": ["llm", "general", "chat"]
            }
        ],
        "endpoints": {"jsonrpc": {"url": jsonrpc_url}}
    }

    return JSONResponse(card)


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# -----------------------------------------------------------------------------
# Heroku entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
