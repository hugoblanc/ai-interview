from fastapi import FastAPI
from pydantic import BaseModel

from agent import AiAgent


app = FastAPI()


class Message(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@app.post("/chat")
async def chat(message: Message) -> Answer:

    agent = AiAgent()
    answer = agent.start_chat(message.question)
    return Answer(answer=answer)
