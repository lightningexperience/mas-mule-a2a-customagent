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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
