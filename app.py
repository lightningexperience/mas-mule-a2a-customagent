# version 9.0.0 - Heroku Optimized (LLM-powered)

import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any

# --- LLM / LangChain Imports (for Groq-powered agent logic) ---
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CustomAgentServer")

app = FastAPI(
    title="Custom Agent A2A Server (JSONRPC Compatible)",
    description="A Python A2A agent designed for JSONRPC transport compatibility."
)

# CRITICAL FOR HEROKU: Add CORS middleware
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
    value: str = Field(..., description="the content string.")

class A2AInput(BaseModel):
    role: str = Field(..., description="Sender role, e.g., 'user'")
    content: List[ContentPart] = Field(..., description="List of content parts")

class A2ATaskRequest(BaseModel):
    taskId: str = Field(..., description="The mandatory unique ID for this task/session.")
    skillId: str = Field(..., description="The skill being invoked (e.g., 'general-llm-query')")
    inputs: List[A2AInput] = Field(..., description="List of messages in the task thread")
    contextId: Optional[str] = Field(None, description="Optional session context ID.")

# --- LLM Configuration (Groq + LangChain) ---

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "llama3-8b-8192")
CONVERSATION_MEMORY_K = 5

# Simple in-memory store for conversation buffers keyed by contextId
MEMORY_STORE: Dict[str, ConversationBufferWindowMemory] = {}

SYSTEM_PROMPT = (
    "You are the Custom Agent, a friendly, fast general-purpose AI assistant. "
    "You are integrated behind an orchestration layer (broker) and must respond "
    "clearly and concisely in plain text. Do not mention routing, A2A, brokers, "
    "or internal implementation details. Just answer the user as helpfully as possible."
)

def get_or_create_memory(conversation_id: str) -> ConversationBufferWindowMemory:
    """Get or create a ConversationBufferWindowMemory for a given conversation ID."""
    if conversation_id not in MEMORY_STORE:
        MEMORY_STORE[conversation_id] = ConversationBufferWindowMemory(
            k=CONVERSATION_MEMORY_K,
            memory_key="chat_history",
            return_messages=True,
        )
        logger.info(f"Created new memory buffer for conversation_id={conversation_id}")
    return MEMORY_STORE[conversation_id]

def run_llm(latest_message_content: str, conversation_id: str) -> str:
    """
    Run the Groq LLM (via LangChain) with conversation memory and return the response text.
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not found in environment variables.")
        return (
            "I'm currently not configured with an LLM backend "
            "(missing GROQ_API_KEY). Please contact the administrator."
        )

    memory = get_or_create_memory(conversation_id)

    # Build prompt template with system message + history + latest human input
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    groq_chat = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
    )

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    logger.info(f"Calling Groq LLM model={GROQ_MODEL_NAME} for conversation_id={conversation_id}")
    response = conversation.predict(human_input=latest_message_content)
    return response


# -----------------------------
#   Agent Card (HEROKU COMPATIBLE)
# -----------------------------

@app.get("/.well-known/agent-card.json")
def get_agent_card(request: Request):
    """
    Returns the agent card with proper JSONRPC transport configuration.
    HEROKU SPECIFIC: Handles both HTTP and HTTPS, uses request.url to get proper scheme
    """
    
    # CRITICAL FOR HEROKU: Use the actual request URL to get the correct base URL
    # Heroku uses HTTPS but may proxy as HTTP, so we need to check headers
    forwarded_proto = request.headers.get("x-forwarded-proto", "http")
    host = request.headers.get("host", str(request.url).split("//")[1].split("/")[0])
    base_url = f"{forwarded_proto}://{host}"
    
    # Enhanced debugging
    logger.info("="*60)
    logger.info("AGENT CARD REQUEST")
    logger.info("="*60)
    logger.info(f"Client: {request.client}")
    logger.info(f"Detected base URL: {base_url}")
    logger.info(f"x-forwarded-proto: {forwarded_proto}")
    logger.info(f"host: {host}")
    logger.info(f"request.url: {request.url}")
    logger.info(f"request.base_url: {request.base_url}")
    logger.info("="*60)
    
    agent_card = {
        "protocolVersion": "0.3.0",
        
        "name": "Custom Agent A2A",
        "description": "General purpose LLM queries with JSONRPC support.",
        "url": f"{base_url}/",
        "version": "9.0.0",
        
        # Optional vendor information
        "vendor": "Custom",
        "apiVersion": "1.0.0",
        
        # Inspector-compatible capabilities
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
                "version": "2.0",
                "contentTypes": ["application/json"]
            }
        }
    }
    
    logger.info(f"Returning agent card with JSONRPC URL: {base_url}/json-rpc")
    
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
    
    logger.info(f"JSONRPC REQUEST received")
    logger.info(f"Payload: {payload}")
    
    # Validate JSONRPC structure
    if "jsonrpc" not in payload or payload["jsonrpc"] != "2.0":
        logger.warning(f"Invalid JSONRPC version: {payload.get('jsonrpc')}")
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
        logger.warning(f"Unknown method: {method}")
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
        logger.info(f"Task request validated: {task_request.taskId}")
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
        logger.info(f"Task {task_request.taskId} completed successfully")
        
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": response
        }
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}", exc_info=True)
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
    logger.info(f"Direct HTTP task request: {task_request.taskId}")
    return await handle_a2a_task(task_request)


# ------------------------------------------------------
#   Core Task Handler Logic (LLM-powered, no routing)
# ------------------------------------------------------

async def handle_a2a_task(task_request: A2ATaskRequest):
    """
    Core logic for handling A2A tasks.
    Called by both JSONRPC and direct HTTP endpoints.

    This version is purely LLM-powered:
    - No keyword routing
    - No support-case detection
    - Mule Fabric / broker is responsible for orchestration
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
    
    # Use contextId if present, otherwise fall back to taskId
    conversation_id = task_request.contextId or task_id

    try:
        agent_response_text = run_llm(latest_message_content, conversation_id)
    except Exception as e:
        logger.error(f"Error calling LLM backend: {str(e)}", exc_info=True)
        agent_response_text = (
            "Something went wrong while contacting the language model. "
            "Please try again later or contact the administrator."
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
                "contextId": conversation_id
            }
        ]
    }


# ------------------------------------------------------
#   Health Check Endpoint (Important for Heroku)
# ------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint - Heroku uses this to verify the app is running"""
    return {
        "status": "healthy",
        "version": "9.0.0",
        "service": "Custom Agent A2A Server",
        "groq_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL_NAME,
    }


# ------------------------------------------------------
#   Root Endpoint
# ------------------------------------------------------

@app.get("/")
def root(request: Request):
    """Root endpoint with service information"""
    forwarded_proto = request.headers.get("x-forwarded-proto", "http")
    host = request.headers.get("host", "localhost")
    base_url = f"{forwarded_proto}://{host}"
    
    return {
        "service": "Custom Agent A2A Server",
        "version": "9.0.0",
        "status": "running",
        "deployment": "Heroku",
        "base_url": base_url,
        "endpoints": {
            "agent_card": f"{base_url}/.well-known/agent-card.json",
            "jsonrpc": f"{base_url}/json-rpc",
            "tasks": f"{base_url}/tasks",
            "health": f"{base_url}/health"
        },
        "instructions": "Use the agent_card URL in A2A Inspector or Mule Fabric to connect"
    }

@app.post("/")
async def root_post_handler(request: Request):
    import uuid  # kept local, as in your original file
    
    body = await request.json()
    logger.info(f"Root POST raw payload: {body}")

    # Case 1: Already a proper A2A TaskRequest → process normally
    try:
        task_request = A2ATaskRequest(**body)
        result = await handle_a2a_task(task_request)
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "kind": "message",
                "role": "agent",
                "messageId": str(uuid.uuid4()),
                "parts": [
                    {
                        "kind": "text",
                        "text": result["outputs"][0]["parts"][0]["text"]
                    }
                ]
            }
        }
    except Exception:
        pass  # Not task-mode → continue

    # Case 2: Fabric "message/send" envelope
    if body.get("method") == "message/send" and "params" in body:
        msg = body["params"]["message"]
        text = msg["parts"][0]["text"]

        task_request = A2ATaskRequest(
            taskId=str(uuid.uuid4()),
            skillId="general-llm-query",
            inputs=[
                A2AInput(
                    role=msg.get("role", "user"),
                    content=[ContentPart(type="text/plain", value=text)]
                )
            ],
            contextId=msg.get("contextId")
        )

        logger.info(f"Converted Fabric message → A2ATaskRequest: {task_request}")

        result = await handle_a2a_task(task_request)
        agent_response = result["outputs"][0]["parts"][0]["text"]

        return {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "kind": "message",
                "role": "agent",
                "messageId": str(uuid.uuid4()),
                "parts": [
                    {
                        "kind": "text",
                        "text": agent_response
                    }
                ]
            }
        }

    # Fallback
    return JSONResponse(
        status_code=400,
        content={"error": "Unrecognized payload format", "body": body}
    )

# HEROKU SPECIFIC: Application must bind to the PORT environment variable
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
