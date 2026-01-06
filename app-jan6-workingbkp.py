# app.py v0.6 - A2A-compatible stateless Groq LLM Custom Agent
# Fix: Added User-Agent header to bypass WAF blocking

import os
import json
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse

import urllib.request
import urllib.error

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
# Helper: Call Groq Chat Completions API
# -----------------------------------------------------------------------------
def call_groq_llm(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    # Defaulting to the versatile model if not specified
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        logger.error("GROQ_API_KEY is not set in environment.")
        return "Error: GROQ_API_KEY is not configured on the server."

    url = "https://api.groq.com/openai/v1/chat/completions"
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

    data = json.dumps(payload).encode("utf-8")
    
    # --- FIX APPLIED HERE ---
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "CustomAgentServer/1.0 (Heroku; Python)",  # Required to bypass security blocks
    }
    # ------------------------

    req = urllib.request.Request(url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            resp_json = json.loads(raw)
        logger.info(f"Groq LLM call succeeded using model: {model}")
        return (
            resp_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except urllib.error.HTTPError as e:
        # Improved error logging to see exactly why it failed
        error_body = e.read().decode('utf-8', errors='ignore')
        logger.error(f"Groq HTTP error {e.code}: {e.reason} - Body: {error_body}")
        return f"Sorry, I had trouble contacting the language model (HTTP {e.code})."
    except Exception as e:
        logger.error(f"Groq LLM error: {e}")
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
# Build A2A-compatible message response (IMPORTANT)
# -----------------------------------------------------------------------------
def build_a2a_message_event(reply_text: str) -> Dict[str, Any]:
    """
    This is the FIX. Mule requires:
        result.kind = "message"
        result.role = "agent"
        result.messageId = uuid
        result.parts = [{kind:"text", text:"..."}]
    """
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
# JSON-RPC handler (Fabric sends POST / with message/send)
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
# Root JSON-RPC endpoint (Fabric POST /)
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
