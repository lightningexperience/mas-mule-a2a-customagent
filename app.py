# version 9.0.0 - Clean LLM Agent (No Routing Logic)

import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CustomAgentServer")

app = FastAPI(
    title="Custom Agent A2A Server (JSONRPC Compatible)",
    description="A pure general-purpose LLM-style agent with NO routing logic."
)

# Add CORS for A2A Inspector / Fabric
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- A2A Protocol Models ---

class ContentPart(BaseModel):
    type: str
    value: str

class A2AInput(BaseModel):
    role: str
    content: List[ContentPart]

class A2ATaskRequest(BaseModel):
    taskId: str
    skillId: str
    inputs: List[A2AInput]
    contextId: Optional[str] = None


# -----------------------------
# Agent Card
# -----------------------------

@app.get("/.well-known/agent-card.json")
def get_agent_card(request: Request):

    forwarded_proto = request.headers.get("x-forwarded-proto", "http")
    host = request.headers.get("host", str(request.url).split("//")[1].split("/")[0])
    base_url = f"{forwarded_proto}://{host}"

    return {
        "protocolVersion": "0.3.0",

        "name": "Custom Agent A2A",
        "description": "General-purpose LLM agent with no routing logic.",
        "url": f"{base_url}/",
        "version": "9.0.0",
        "vendor": "Custom",
        "apiVersion": "1.0.0",

        "capabilities": {
            "pushNotifications": False,
            "streaming": False,
            "batching": False,
            "stateful": False
        },

        "securitySchemes": {},

        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],

        "skills": [
            {
                "id": "general-llm-query",
                "name": "General LLM Query",
                "description": "Responds conversationally to general input.",
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
                "examples": [
                    "Explain gravity in simple terms.",
                    "Give me a fun fact about space.",
                    "Summarize any concept."
                ]
            }
        ],

        "preferredTransport": "JSONRPC",

        "transports": {
            "JSONRPC": {
                "url": f"{base_url}/json-rpc",
                "version": "2.0",
                "contentTypes": ["application/json"]
            }
        }
    }


# ------------------------------------------------------
# JSONRPC Endpoint
# ------------------------------------------------------

@app.post("/json-rpc")
async def json_rpc_handler(payload: Dict[str, Any]):

    if payload.get("jsonrpc") != "2.0":
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {"code": -32600, "message": "Invalid JSONRPC version"}
        }

    method = payload.get("method")
    params = payload.get("params")
    rpc_id = payload.get("id")

    if method != "task":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": "Unknown method"}
        }

    try:
        task_request = A2ATaskRequest(**params)
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32602, "message": str(e)}
        }

    result = await handle_a2a_task(task_request)

    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


# ------------------------------------------------------
# Main task handler (NO routing logic)
# ------------------------------------------------------

async def handle_a2a_task(task_request: A2ATaskRequest):
    """
    The Custom Agent NEVER routes.
    It ALWAYS returns a simple general-purpose conversational response.
    """

    task_id = task_request.taskId

    try:
        latest = task_request.inputs[-1].content[-1].value
    except Exception:
        latest = ""

    # Pure conversational response — NO routing, NO rejection, NO classification
    agent_response_text = (
        f"Custom Agent Response: You said: \"{latest}\" — Let me know if you "
        f"want more help with this or something else! (Task ID: {task_id})"
    )

    return {
        "status": "completed",
        "taskId": task_id,
        "outputs": [
            {
                "kind": "message",
                "role": "agent",
                "parts": [
                    {"kind": "text", "text": agent_response_text}
                ],
                "contextId": task_request.contextId or task_id
            }
        ]
    }


# ------------------------------------------------------
# Direct /tasks fallback
# ------------------------------------------------------

@app.post("/tasks")
async def handle_a2a_task_endpoint(task_request: A2ATaskRequest):
    return await handle_a2a_task(task_request)


# ------------------------------------------------------
# Root (GET + POST)
# ------------------------------------------------------

@app.get("/")
def root(request: Request):

    forwarded_proto = request.headers.get("x-forwarded-proto", "http")
    host = request.headers.get("host", "localhost")
    base_url = f"{forwarded_proto}://{host}"

    return {
        "service": "Custom Agent A2A Server",
        "version": "9.0.0",
        "status": "running",
        "deployment": "Heroku",
        "base_url": base_url
    }


@app.post("/")
async def root_post_handler(request: Request):
    """Handles message/send envelopes coming from Mule Fabric."""
    import uuid

    body = await request.json()

    # Try parsing as A2A
    try:
        task_request = A2ATaskRequest(**body)
        result = await handle_a2a_task(task_request)
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": result
        }
    except:
        pass

    # Fabric message/send format
    if body.get("method") == "message/send" and "params" in body:
        msg = body["params"]["message"]
        text = msg["parts"][0]["text"]

        task_request = A2ATaskRequest(
            taskId=str(uuid.uuid4()),
            skillId="general-llm-query",
            inputs=[A2AInput(
                role=msg.get("role", "user"),
                content=[ContentPart(type="text/plain", value=text)]
            )],
            contextId=msg.get("contextId")
        )

        result = await handle_a2a_task(task_request)
        agent_response = result["outputs"][0]["parts"][0]["text"]

        return {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "kind": "message",
                "role": "agent",
                "messageId": str(uuid.uuid4()),
                "parts": [{"kind": "text", "text": agent_response}]
            }
        }

    return JSONResponse(
        status_code=400,
        content={"error": "Unrecognized payload", "body": body}
    )


# ------------------------------------------------------
# Heroku Entrypoint
# ------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
