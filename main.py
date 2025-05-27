import logging
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat import get_org_chain, get_general_chain, clear_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTT-Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Mode(str, Enum):
    org = "org"
    general = "general"

class ChatRequest(BaseModel):
    question: str
    session_id: str = Field("default")
    mode: Mode = Field(Mode.org)

class ChatResponse(BaseModel):
    answer: str
    source_documents: list[str] | None = None  # org เท่านั้น

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        chain = (
            get_org_chain(req.session_id)
            if req.mode == Mode.org
            else get_general_chain(req.session_id)
        )
        result = chain.invoke(
            {"question": req.question},
            config={"configurable": {"session_id": req.session_id}},
        )

        answer = (
            result.get("answer")
            or result.get("output")
            or result.get("text")
            or ""
        )
        if req.mode == Mode.org and not answer.strip():
            answer = (
                "ขอโทษนะคะ ไม่พบข้อมูลขององค์กรที่เกี่ยวข้องกับคำถามนี้ "
                "ลองถามในโหมดทั่วไปดูนะคะ"
            )

        src_docs = (
            [d.page_content for d in result.get("source_documents", [])]
            if req.mode == Mode.org
            else None
        )
        return ChatResponse(answer=answer, source_documents=src_docs)

    except Exception as exc:
        logger.exception("Chat endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/clear")
async def clear_chat(session_id: str = "default"):
    clear_memory(session_id)
    return {"status": "success", "message": f"Cleared session '{session_id}'"}

@app.get("/")
def root():
    return {"message": "TTT-Assistant backend is running"}
