# version 8.0.0 - Fixed for A2A Inspector Compatibility

import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CustomAgentServer")

app = FastAPI(
    title="Custom Agent A2A Server (JSONRPC Compatible)",
    description="A Python A2A agent designed for JSONRPC transport compatibility."
)

# Add CORS middleware to allow A2A Inspector to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    skillId: str = Field(..., description="The skill being invoked (e.g., 'general-llm-query')")
    inputs: List[A2AInput] = Field(..., description="List of messages in the task thread")
    contextId: Optional[str] = Field(None, description="Optional session context ID.")


# -----------------------------
#   Agent Card (JSONRPC FIX)
# -----------------------------

@app.get("/.well-known/agent-card.json")
def get_agent_card(request: Request):
    """
    Returns the agent card with proper JSONRPC transport configuration.
    This is the critical endpoint that A2A Inspector uses to discover the agent.
    """
    
    base_url = str(request.base_url).rstrip('/')
    
    # Ensure we're returning the exact structure A2A Inspector expects
    agent_card = {
        "protocolVersion": "0.3.0",
        
        "name": "Custom Agent A2A",
        "description": "General purpose LLM queries with JSONRPC support.",
        "url": f"{base_url}/",
        "version": "8.0.0",
        
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
                    "Tell me a joke about Python.",
                    "Explain quantum computing in simple terms."
                ],
                "tags": ["general", "llm", "knowledge"]
            }
        ],
        
        # CRITICAL: A2A Inspector requires JSONRPC transport
        "preferredTransport": "JSONRPC",
        
        "transports": {
            "JSONRPC": {
                "url": f"{base_url}/json-rpc",
                "version": "2.0"
            }
        }
    }
    
    logger.info(f"Agent card requested. Base URL: {base_url}")
    logger.info(f"JSONRPC endpoint: {base_url}/json-rpc")
    
    return agent_card


# ------------------------------------------------------
#   JSONRPC ENDPOINT (REQUIRED BY INSPECTOR)
# ------------------------------------------------------

@app.post("/json-rpc")
async def json_rpc_handler(payload: Dict[str, Any]):
    """
    JSONRPC 2.0 endpoint to satisfy A2A Inspector requirements.
    Routes method "task" to the task handler.
    """
    
    logger.info(f"JSONRPC REQUEST: {payload}")
    
    # Validate JSONRPC structure
    if "jsonrpc" not in payload or payload["jsonrpc"] != "2.0":
        return {
            "jsonrpc": "2.0",
            "id": payload.get("id"),
            "error": {
                "code": -32600,
                "message": "Invalid Request: jsonrpc version must be 2.0"
            }
        }
    
    method = payload.get("method")
    params = payload.get("params")
    rpc_id = payload.get("id")
    
    # Only support "task" method
    if method != "task":
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }
    
    # Validate and parse task parameters
    try:
        task_request = A2ATaskRequest(**params)
    except ValidationError as e:
        logger.error(f"Invalid task parameters: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {
                "code": -32602,
                "message": f"Invalid params: {str(e)}"
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error parsing params: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }
    
    # Process the task
    try:
        response = await handle_a2a_task(task_request)
        
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": response
        }
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}")
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {
                "code": -32603,
                "message": f"Internal error processing task: {str(e)}"
            }
        }


# ------------------------------------------------------
#   Main A2A Task Endpoint (Direct HTTP fallback)
# ------------------------------------------------------

@app.post("/tasks")
async def handle_a2a_task_endpoint(task_request: A2ATaskRequest):
    """
    Direct HTTP endpoint for task processing (fallback).
    Most A2A clients will use JSONRPC, but this provides compatibility.
    """
    return await handle_a2a_task(task_request)


# ------------------------------------------------------
#   Core Task Handler Logic
# ------------------------------------------------------

async def handle_a2a_task(task_request: A2ATaskRequest):
    """
    Core logic for handling A2A tasks.
    Called by both JSONRPC and direct HTTP endpoints.
    """
    
    task_id = task_request.taskId
    skill_id = task_request.skillId
    
    logger.info(f"Processing Task ID: {task_id}, Skill ID: {skill_id}")
    
    # Extract the latest message
    try:
        latest_message_content = task_request.inputs[-1].content[-1].value
    except (IndexError, AttributeError) as e:
        error_msg = "Invalid or missing message inputs in A2A payload."
        logger.error(f"{error_msg}: {str(e)}")
        return {
            "status": "failed",
            "taskId": task_id,
            "error": error_msg
        }
    
    logger.info(f"Received Message: {latest_message_content[:80]}...")
    
    # Simple routing logic based on message content
    if "support case" in latest_message_content.lower() or "ticket" in latest_message_content.lower():
        agent_response_text = (
            "I am the Custom Agent (General LLM). "
            "I cannot handle Salesforce support cases; please route to Agentforce."
        )
    else:
        agent_response_text = (
            f"Custom Agent Response: I've analyzed your message '{latest_message_content[:50]}...'. "
            f"This is a demonstration of A2A protocol integration. (Task ID: {task_id})"
        )
    
    # Return properly formatted A2A response
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
                "contextId": task_request.contextId or task_id
            }
        ]
    }


# ------------------------------------------------------
#   Health Check Endpoint
# ------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "version": "8.0.0",
        "service": "Custom Agent A2A Server"
    }


# ------------------------------------------------------
#   Root Endpoint
# ------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint with service information"""
    return {
        "service": "Custom Agent A2A Server",
        "version": "8.0.0",
        "status": "running",
        "endpoints": {
            "agent_card": "/.well-known/agent-card.json",
            "jsonrpc": "/json-rpc",
            "tasks": "/tasks",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
