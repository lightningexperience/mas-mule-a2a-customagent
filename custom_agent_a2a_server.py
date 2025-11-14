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
    title="Custom Agent A2A Server (Fix)",
    description="A Python A2A agent designed to handle context echoing for MuleSoft Broker stability."
)

# --- A2A Protocol Schemas (Strictly Enforced) ---

class ContentPart(BaseModel):
    type: str = Field(..., description="MIME type, e.g., 'text/plain'")
    value: str = Field(..., description="The content string.")

class A2AInput(BaseModel):
    role: str = Field(..., description="Sender role, e.g., 'user'")
    content: List[ContentPart] = Field(..., description="List of content parts")

class A2ATaskRequest(BaseModel):
    # CRITICAL: This is the field (pk) that MUST NOT BE NULL in the response.
    taskId: str = Field(..., description="The mandatory unique ID for this task/session.")
    skillId: str = Field(..., description="The skill being invoked (e.g., 'broker-orchestrator')")
    inputs: List[A2AInput] = Field(..., description="List of messages in the task thread")
    contextId: Optional[str] = Field(None, description="Optional session context ID.")


# --- Health Check and Discovery (FINAL FIX APPLIED HERE) ---

@app.get("/.well-known/agent-card.json")
def get_agent_card(request: Request):
    """
    Returns the valid Agent Card for discovery. 
    Includes all mandatory fields (examples, tags) and uses standard FastAPI JSON return.
    """
    base_url = str(request.base_url).rstrip('/')
    
    # We return the dict directly and let FastAPI handle JSONResponse
    return {
        "protocolVersion": "0.3.0",
        "name": "Custom Agent A2A",
        "description": "General purpose LLM queries.",
        "url": f"{base_url}/",
        "version": "6.0.0",
        "capabilities": {"pushNotifications": False, "streaming": False},
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
                # --- MANDATORY FIXES APPLIED HERE ---
                "examples": ["What is the capital of France?", "Tell me a joke about Python."],
                "tags": ["general", "llm", "knowledge"] 
            }
        ],
        "transports": {
            "http": {
                "url": f"{base_url}/"
            }
        }
    }

# --- Main A2A Task Endpoint (NPE Fix) ---

@app.post("/tasks")
async def handle_a2a_task(task_request: A2ATaskRequest):
    """
    Handles the A2A task request, extracts the mandatory taskId, and returns
    a protocol-compliant response using the v0.3.0 structure.
    """
    
    # 1. CRITICAL CONTEXT EXTRACTION (Fixes the Broker's NPE on pk)
    task_id = task_request.taskId
    
    try:
        latest_message_content = task_request.inputs[-1].content[-1].value
    except Exception:
        # Handle cases where inputs list might be malformed or empty
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

    # --- 2. CORE AGENT LOGIC (Mocked) ---
    if "support case" in latest_message_content.lower() or "ticket" in latest_message_content.lower():
        agent_response_text = "I am the Custom Agent (General LLM). I cannot handle specific Salesforce support cases; please ask the Orchestrator to route you to the appropriate Agentforce."
    else:
        # Placeholder for actual LLM call (e.g., Groq, Azure OpenAI)
        agent_response_text = f"Custom Agent (Groq): Analyzing '{latest_message_content[:50]}...'. Continuous learning is essential. (Task ID: {task_id})"

    # --- 3. CONSTRUCTING THE A2A RESPONSE (V0.3.0 Protocol Compliance) ---
    
    return {
        "status": "completed",
        "taskId": task_id, 
        "outputs": [
            {
                "kind": "message",
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": agent_response_text
                    }
                ],
                "contextId": task_id
            }
        ]
    }
