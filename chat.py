from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableWithMessageHistory

# ---- compatibility import สำหรับ LC 0.1 / 0.2 ----
try:
    from langchain_core.retrievers import BaseRetriever
except ImportError:  # LC < 0.2
    from langchain.retrievers import BaseRetriever  # type: ignore

from pydantic import ConfigDict

# ─────────────────── logging ───────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────── Ollama wrapper ───────────────────
class LocalLLM(LLM):
    api_url: str
    model_name: str

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "local_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:  # type: ignore[override]
        return {"model": self.model_name, "api_url": self.api_url}

    def _call(  # type: ignore[override]
        self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> str:
        resp = requests.post(
            f"{self.api_url}/api/generate",
            json={"model": self.model_name, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]

# ─────────────────── Prompt templates ───────────────────
ORG_PROMPT = PromptTemplate(
    template="""
บทสนทนาที่ผ่านมา
{chat_history}

ข้อมูลอ้างอิงจากองค์กร
{context}

คำถาม
{question}

จงตอบโดยอิงข้อมูลข้างต้น  หากข้อมูลอยู่ใน chat_history ก็ใช้ได้
ถ้าหาข้อมูลไม่ได้ทั้งใน context และ chat_history ให้ตอบว่า
“ขอโทษนะคะ ไม่พบข้อมูลขององค์กรที่เกี่ยวข้องกับคำถามนี้ ลองถามในโหมดทั่วไปดูนะคะ”
""",
    input_variables=["chat_history", "context", "question"],
)

CONDENSE_PROMPT = PromptTemplate(
    template="""
สรุปหรือเติมเต็มคำถามต่อไปนี้ให้ชัดเจน (ภาษาไทย) โดยอ้างอิงบทสนทนา

บทสนทนา:
{chat_history}

คำถามเดิม:
{question}

คำถามที่สรุปแล้ว:
""",
    input_variables=["chat_history", "question"],
)

GENERAL_PROMPT = PromptTemplate(
    template="{chat_history}\nผู้ใช้: {question}\nผู้ช่วย:",
    input_variables=["chat_history", "question"],
)

# ───────────── caches / paths / constants ─────────────
_message_histories: Dict[str, ChatMessageHistory] = {}
_org_chain_cache: Dict[str, RunnableWithMessageHistory] = {}
_general_chain_cache: Dict[str, RunnableWithMessageHistory] = {}

FAISS_PATH = "faiss_index"
EMBED_MODEL = "local_models/bge-m3"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "TTT-Assistant-4b:latest"

# ─────────────────── helpers ───────────────────
def _llm() -> LocalLLM:
    return LocalLLM(api_url=OLLAMA_URL, model_name=MODEL_NAME)


def _history(sess: str) -> ChatMessageHistory:
    return _message_histories.setdefault(sess, ChatMessageHistory())

# ──────────────── FallbackRetriever ────────────────
class FallbackRetriever(BaseRetriever):
    """
    ถ้า base_retriever หาเอกสารไม่พบ → คืน Document เปล่า 1 ชิ้น
    เพื่อไม่ให้ `context` ว่าง (LLM จะยังเห็น chat_history)
    """

    base_retriever: Any
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # sync
    def _get_relevant_documents(self, query: str):
        docs = self.base_retriever.get_relevant_documents(query)
        return docs or [Document(page_content="")]

    # async
    async def _aget_relevant_documents(self, query: str):
        docs = await self.base_retriever.aget_relevant_documents(query)
        return docs or [Document(page_content="")]

# ─────────────────── Chains ───────────────────
def get_org_chain(session_id: str) -> RunnableWithMessageHistory:
    if session_id in _org_chain_cache:
        return _org_chain_cache[session_id]

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = FallbackRetriever(
        base_retriever=store.as_retriever(search_kwargs={"k": 3})
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=_llm(),
        retriever=retriever,
        condense_question_llm=_llm(),  # for LC < 0.2
        combine_docs_chain_kwargs={"prompt": ORG_PROMPT},
        return_source_documents=True,
        verbose=True,
    )
    qa.question_generator.prompt = CONDENSE_PROMPT  # type: ignore[attr-defined]

    wrapped = RunnableWithMessageHistory(
        qa,
        lambda sid: _history(sid),
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    _org_chain_cache[session_id] = wrapped
    return wrapped


def get_general_chain(session_id: str) -> RunnableWithMessageHistory:
    if session_id in _general_chain_cache:
        return _general_chain_cache[session_id]

    llm_chain = LLMChain(llm=_llm(), prompt=GENERAL_PROMPT, output_key="answer")

    wrapped = RunnableWithMessageHistory(
        llm_chain,
        lambda sid: _history(sid),
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    _general_chain_cache[session_id] = wrapped
    return wrapped

# ─────────────────── utilities ───────────────────
def clear_memory(session_id: str):
    _message_histories.pop(session_id, None)
    _org_chain_cache.pop(session_id, None)
    _general_chain_cache.pop(session_id, None)
