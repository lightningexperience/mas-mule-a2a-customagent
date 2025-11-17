# app.py v0.4 - A2A-compatible stateless Groq LLM Custom Agent (no LangChain)

import os
import json
import logging
from typing import Any, Dict, List, Optional

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
    """
    Calls the Groq Chat Completions API using environment variables:
    - GROQ_API_KEY
    - GROQ_MODEL (e.g., "llama-3.3-70b-versatile")

    Returns plain text from the model or a simple error string.
    """
    api_key = os.environ.get("GROQ_API_KEY")
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
                    "You are a helpful, general-purpose AI assistant. "
                    "Return concise, direct answers in plain text. "
                    "Do not add routing labels or meta commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            resp_json = json.loads(raw)
        logger.info("Groq LLM call succeeded.")
        return (
            resp_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except urllib.error.HTTPError as e:
        logger.error(f"Groq HTTP error: {e} - {e.read().decode('utf-8', errors='ignore')}")
        return "Sorry, I had trouble contacting the language model (HTTP error)."
    except Exception as e:
        logger.error(f"Groq LLM error: {e}")
        return "Sorry, I had trouble contacting the language model."


# -----------------------------------------------------------------------------
# Helper: Extract plain text from Fabric/A2A-style "message" payload
# -----------------------------------------------------------------------------
def extract_text_from_message(message: Dict[str, Any]) -> str:
    """
    The inspector / Fabric sends messages like:
    {
      "role": "user",
      "parts": [
        { "text": "hey", "kind": "text" }
      ],
      "messageId": "...",
      "kind": "message"
    }
    We simply join all 'text' fields we see.
    """
    parts: List[Dict[str, Any]] = message.get("parts", [])
    texts: List[str] = []
    for p in parts:
        # Two common patterns:
        #  - { "text": "...", "kind": "text" }
        #  - { "value": "...", "type": "text/plain" } (A2A style)
        if "text" in p:
            texts.append(str(p["text"]))
        elif p.get("type") == "text/plain" and "value" in p:
            texts.append(str(p["value"]))
    joined = "\n".join(texts).strip()
    logger.info(f"CustomAgentServer:Received Message: {joined[:80]}...")
    return joined


# -----------------------------------------------------------------------------
# JSON-RPC handler
# -----------------------------------------------------------------------------
async def handle_jsonrpc(payload: Dict[str, Any]) -> JSONResponse:
    logger.info(f"CustomAgentServer:Root POST raw payload: {payload}")

    jsonrpc_version = payload.get("jsonrpc")
    method = payload.get("method")
    msg_id = payload.get("id")

    if jsonrpc_version != "2.0":
        error = {"code": -32600, "message": "Invalid JSON-RPC version"}
        return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": error}, status_code=400)

    if method != "message/send":
        error = {"code": -32601, "message": f"Unsupported method: {method}"}
        return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": error}, status_code=400)

    # Extract user text from params.message
    params = payload.get("params", {})
    message = params.get("message", {})
    user_text = extract_text_from_message(message)

    # Call Groq LLM
    reply_text = call_groq_llm(user_text)

    # Build assistant message in A2A/Fabric-friendly format
    assistant_message = {
        "role": "assistant",
        "parts": [
            {
                "text": reply_text,
                "kind": "text",
            }
        ],
        "kind": "message",
    }

    response_body = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "message": assistant_message
        },
    }
    return JSONResponse(response_body)


# -----------------------------------------------------------------------------
# A2A /tasks endpoint
# -----------------------------------------------------------------------------
@app.post("/tasks")
async def tasks_endpoint(task_request: Dict[str, Any] = Body(...)) -> JSONResponse:
    """
    Minimal A2A-compatible /tasks implementation.

    Expected shape (simplified):

    {
      "taskId": "123",
      "skillId": "general-llm-query",
      "inputs": [
        {
          "role": "user",
          "content": [
            {"type": "text/plain", "value": "hello"}
          ]
        }
      ],
      "contextId": null
    }
    """
    task_id = task_request.get("taskId")
    skill_id = task_request.get("skillId")
    inputs = task_request.get("inputs", [])

    logger.info(
        f"CustomAgentServer:Processing Task ID: {task_id}, Skill ID: {skill_id}"
    )

    # Extract plain text from inputs
    all_texts: List[str] = []
    for inp in inputs:
        content_items = inp.get("content", [])
        for c in content_items:
            if c.get("type") == "text/plain" and "value" in c:
                all_texts.append(str(c["value"]))

    user_text = "\n".join(all_texts).strip()
    logger.info(f"CustomAgentServer:Task received text: {user_text[:80]}...")

    reply_text = call_groq_llm(user_text)

    task_response = {
        "taskId": task_id,
        "status": "completed",
        "outputs": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text/plain",
                        "value": reply_text,
                    }
                ],
            }
        ],
    }
    return JSONResponse(task_response)


# -----------------------------------------------------------------------------
# Root JSON-RPC endpoint (Fabric calls POST / with JSON-RPC payload)
# -----------------------------------------------------------------------------
@app.post("/")
async def root_jsonrpc_endpoint(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    return await handle_jsonrpc(payload)


# -----------------------------------------------------------------------------
# Dedicated JSON-RPC endpoint (Inspector / A2A clients can use /json-rpc)
# -----------------------------------------------------------------------------
@app.post("/json-rpc")
async def jsonrpc_endpoint(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    return await handle_jsonrpc(payload)


# -----------------------------------------------------------------------------
# Agent card endpoint: /.well-known/agent-card.json
# -----------------------------------------------------------------------------
@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request) -> JSONResponse:
    logger.info("============================================================")
    logger.info("CustomAgentServer:AGENT CARD REQUEST")
    logger.info("============================================================")

    client = request.client
    logger.info(f"CustomAgentServer:Client: {client}")

    # Derive base URL from request
    base_url = str(request.base_url).rstrip("/")
    logger.info(f"CustomAgentServer:Detected base URL: {base_url}")

    xfp = request.headers.get("x-forwarded-proto")
    host = request.headers.get("host")
    logger.info(f"CustomAgentServer:x-forwarded-proto: {xfp}")
    logger.info(f"CustomAgentServer:host: {host}")
    logger.info(f"CustomAgentServer:request.url: {request.url}")
    logger.info(f"CustomAgentServer:request.base_url: {request.base_url}")
    logger.info("============================================================")

    jsonrpc_url = f"{base_url}/json-rpc"
    logger.info(f"CustomAgentServer:Returning agent card with JSONRPC URL: {jsonrpc_url}")

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
                "examples": [
                    "hey",
                    "whats the capital of india",
                    "Tell me a fun fact about space",
                ],
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
            }
        ],
        # Some inspectors/clients expect explicit endpoint info
        "endpoints": {
            "jsonrpc": {
                "url": jsonrpc_url
            }
        },
    }

    return JSONResponse(card)
