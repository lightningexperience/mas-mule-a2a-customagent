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

@app.get("/.well-known/agent-card.json")
def agent_card():
    base_url = os.getenv("BASE_URL", "http://localhost:9000")
    return {
        "name": "CustomAgent A2A",
        "description": "Handles general queries using Groq LLM.",
        "url": base_url,
        "version": "1.0",
        "protocolVersion": "0.3.0",
        "preferredTransport": "HTTP+JSON",
        "additionalInterfaces": [
            {
                "url": base_url,
                "transport": "HTTP+JSON"
            }
        ],
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [
            {
                "id": "general-assistance",
                "name": "General Assistance",
                "description": "Responds to general-purpose queries using a fast LLM.",
                "tags": ["groq", "llm", "general"]
            }
        ]
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
