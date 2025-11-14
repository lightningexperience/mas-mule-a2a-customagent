# version 7.0.0

import os
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CustomAgentServer")

app = FastAPI(
    title="Custom Agent A2A Server (JSONRPC Compatible)",
    description="A Python A2A agent designed for JSONRPC transport compatibility."
)

# --- A2A Protocol Schemas (Strictly Enforced) ---

class ContentPart(BaseModel):
    type: str = Field(..., description="MIME type, e.g., 'text/plain'")
    value: str = Field(..., description="The content string.")

class A2AInput(BaseModel):
    role: str = Field(..., description="Sender role, e.g., 'user'")
    content: List[ContentPart] = Field(..., description="List of content parts")

class A2ATaskRequest(BaseModel):
    taskId: str = Field(..., description="The mandatory unique ID for this task/session.")
    skillId: str = Field(..., description="The skill being invoked (e.g., 'broker-orchestrator')")
    inputs: List[A2AInput] = Field(..., description="List of messages in the task thread")
    contextId: Optional[str] = Field(None, description="Optional session context ID.")



# -----------------------------
#   Agent Card (JSONRPC FIX)
# -----------------------------

@app.get("/.well-known/agent-card.json")
def get_agent_card(request: Request):

    base_url = str(request.base_url).rstrip('/')

    return {
        "protocolVersion": "0.3.0",

        # Add a version marker
        "debug": "VERSION-XYZ-123",

        "name": "Custom Agent A2A",
        "description": "General purpose LLM queries - VERSION-XYZ-123.",
        "url": f"{base_url}/",
        "version": "6.0.0",

        # Inspector-compatible capabilities
        "capabilities": {
            "pushNotifications": False,
            "streaming": False
        },

        "securitySchemes": {},

        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],

        "skills": [
            {
                "id": "general-llm-query",
                "name": "General LLM Query",
                "description": "Answers general knowledge and LLM questions.",
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
                "examples": [
                    "What is the capital of France?",
                    "Tell me a joke about Python."
                ],
                "tags": ["general", "llm", "knowledge"]
            }
        ],

        # IMPORTANT: A2A Inspector requires JSONRPC transport
        "preferredTransport": "JSONRPC",

        "transports": {
            "JSONRPC": {
                "url": f"{base_url}/json-rpc"
            }
        }
    }



# ------------------------------------------------------
#   MINIMAL JSONRPC ENDPOINT (REQUIRED BY INSPECTOR)
# ------------------------------------------------------

@app.post("/json-rpc")
async def json_rpc_handler(payload: Dict[str, Any]):

    """
    Minimal JSONRPC endpoint to satisfy A2A Inspector.
    We do NOT implement full JSONRPC 2.0.
    We only route method "task" → /tasks
    """

    logger.info(f"JSONRPC REQUEST: {payload}")

    method = payload.get("method")
    params = payload.get("params")
    rpc_id = payload.get("id")

    if method != "task":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": "Method not found"}
        }

    # JSONRPC → /tasks mapping
    try:
        task_request = A2ATaskRequest(**params)
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32602, "message": f"Invalid params: {str(e)}"}
        }

    # Call the existing task handler
    response = await handle_a2a_task(task_request)

    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": response
    }




# ------------------------------------------------------
#   Main A2A Task Endpoint (Your Existing Logic)
# ------------------------------------------------------

@app.post("/tasks")
async def handle_a2a_task(task_request: A2ATaskRequest):

    task_id = task_request.taskId

    try:
        latest_message_content = task_request.inputs[-1].content[-1].value
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "failed",
                "taskId": task_id,
                "error": "Invalid or missing message inputs in A2A payload."
            }
        )

    logger.info(f"Received Task ID: {task_id}")
    logger.info(f"Received Message: {latest_message_content[:80]}...")

    # Your logic
    if "support case" in latest_message_content.lower() or "ticket" in latest_message_content.lower():
        agent_response_text = (
            "I am the Custom Agent (General LLM). "
            "I cannot handle Salesforce support cases; route me to Agentforce."
        )
    else:
        agent_response_text = (
            f"Custom Agent (Groq): Analyzing '{latest_message_content[:50]}...'. "
            f"(Task ID: {task_id})"
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
                "contextId": task_id
            }
        ]
    }
