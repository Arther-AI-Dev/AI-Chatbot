from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import initialize_qa_chain, clear_memory
import logging
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")


app = FastAPI()

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
    session_id: str = "default"

@app.on_event("startup")
async def startup_event():
    global qa_chain
    qa_chain = initialize_qa_chain()

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        qa_chain = initialize_qa_chain(request.session_id)
        result = qa_chain.invoke(
            {"question": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # Log the result for debugging
        logging.info(f"Chat result: {result}")
        
        # Handle potential missing fields gracefully
        answer = result.get("answer", "")
        if not answer and isinstance(result, dict):
            # Try to get answer from output field
            answer = result.get("output", "No answer generated")
            
        source_docs = result.get("source_documents", [])
        if isinstance(source_docs, list):
            source_contents = [doc.page_content for doc in source_docs]
        else:
            source_contents = []
            
        return {
            "answer": answer,
            "source_documents": source_contents
        }
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_chat(session_id: str = "default"):
    clear_memory(session_id)
    return {"status": "success", "message": f"Cleared chat history for session {session_id}"}

@app.get("/")
def read_root():
    return {"message": "Welcome AI Chatbot is running"}