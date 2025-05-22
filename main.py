from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import initialize_qa_chain

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain = None

class ChatRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    global qa_chain
    qa_chain = initialize_qa_chain()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = qa_chain({"question": request.question})
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))