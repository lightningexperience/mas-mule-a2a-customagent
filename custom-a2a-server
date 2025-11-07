# --- CustomAgent A2A Server ---
# File: custom_agent_a2a_server.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from uuid import uuid4

app = FastAPI()

memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history", return_messages=True)

class Part(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    parts: list[Part]
    messageId: str

class TaskRequest(BaseModel):
    id: str
    message: Message

@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name": "CustomAgent A2A",
        "description": "Handles general queries using Groq LLM.",
        "url": os.getenv("BASE_URL", "http://localhost:9000"),
        "version": "1.0",
        "capabilities": {"streaming": False},
    }

@app.post("/tasks/send")
def handle_task(task: TaskRequest):
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are CustomAgent, a helpful chatbot that assists users with general inquiries."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ])

    conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)
    user_msg = task.message.parts[0].text
    agent_reply = conversation.predict(human_input=user_msg)

    return {
        "id": task.id,
        "status": {"state": "completed"},
        "messages": [
            task.message.dict(),
            {"role": "agent", "parts": [{"text": agent_reply}]},
        ]
    }


# --- Agentforce A2A Server ---
# File: agentforce_a2a_server.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
import requests
import uuid

app = FastAPI()

class Part(BaseModel):
    text: str

class Message(BaseModel):
    role: str
    parts: list[Part]
    messageId: str

class TaskRequest(BaseModel):
    id: str
    message: Message

@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name": "Agentforce A2A",
        "description": "Escalates to Salesforce AI support agent.",
        "url": os.getenv("BASE_URL", "http://localhost:9001"),
        "version": "1.0",
        "capabilities": {"streaming": False},
    }

@app.post("/tasks/send")
def handle_task(task: TaskRequest):
    sf_instance = os.getenv("SF_INSTANCE")
    url = f"{sf_instance}/services/oauth2/token"
    token_response = requests.post(url, data={
        'grant_type': 'client_credentials',
        'client_id': os.getenv("SF_CLIENT_ID"),
        'client_secret': os.getenv("SF_CLIENT_SECRET")
    })
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]

    session_payload = {
        "externalSessionKey": str(uuid.uuid4()),
        "instanceConfig": {"endpoint": sf_instance},
        "streamingCapabilities": {"chunkTypes": ["Text"]},
        "bypassUser": True
    }
    session_res = requests.post(
        f"https://api.salesforce.com/einstein/ai-agent/v1/agents/0XxWt0000005qu1KAA/sessions",
        json=session_payload,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    )
    session_id = session_res.json()["sessionId"]

    message_payload = {
        "message": {
            "sequenceId": 1,
            "type": "Text",
            "text": task.message.parts[0].text
        }
    }
    response = requests.post(
        f"https://api.salesforce.com/einstein/ai-agent/v1/sessions/{session_id}/messages/stream",
        json=message_payload,
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        stream=True
    )
    final_msg = ""
    for line in response.iter_lines():
        if line and line.decode("utf-8").startswith("data: "):
            try:
                event = json.loads(line.decode("utf-8")[6:])
                if event.get("message", {}).get("type") in ["TextChunk", "Inform"]:
                    final_msg = event["message"]["message"]
                    break
            except:
                continue

    return {
        "id": task.id,
        "status": {"state": "completed"},
        "messages": [
            task.message.dict(),
            {"role": "agent", "parts": [{"text": final_msg}]},
        ]
    }
