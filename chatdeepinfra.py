import requests
from typing import Optional, List, Dict, Any
from langchain.llms.base import LLM
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableWithMessageHistory
import os
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------
# 1. Define LocalLLM with pydantic fields
# ----------------------------------------
class LocalLLM(LLM):
    api_key: str
    model_name: str = "google/gemma-3-27b-it"

    @property
    def _llm_type(self) -> str:
        return "deepinfra_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model_name}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        # Log the incoming prompt
        logger.info(f"Prompt received: {prompt}")
        logger.info(f"Additional kwargs: {kwargs}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        res = requests.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            headers=headers,
            json=payload,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    


# ----------------------------------------
# 2. Define a PromptTemplate including chat history
# ----------------------------------------
custom_template = """ต่อไปนี้คุณชื่อ TTT-Assistant ผู้ช่วย AI ของบริษัท TTT Brothers Co., Ltd.
\
บทสนทนาที่ผ่านมา:{chat_history}
ข้อมูลที่มี:{context}
\
คำถาม:{question}
ตอบกลับเป็นภาษาไทย"""

CUSTOM_PROMPT = PromptTemplate(
    template=custom_template,
    input_variables=["chat_history", "context", "question"]
)

# Add global variable for QA chain
_message_histories = {}
_qa_chain_dict = {}

FAISS_INDEX_PATH = "faiss_index"
LOCAL_MODEL_PATH = "local_models/bge-m3"

def get_message_history(session_id: str) -> ChatMessageHistory:
    """Get or create message history for a specific session"""
    if session_id not in _message_histories:
        _message_histories[session_id] = ChatMessageHistory()
    return _message_histories[session_id]

def initialize_qa_chain(session_id: str = "default"):
    """Initialize or get existing QA chain for a session"""
    if session_id in _qa_chain_dict:
        return _qa_chain_dict[session_id]
        
    API_KEY = "iaLJrht7O6tDk1KOn6lyE20RUD6Nof7U"  # Consider moving this to environment variable
    MODEL_NAME = "google/gemma-3-27b-it"
    llm = LocalLLM(api_key=API_KEY, model_name=MODEL_NAME)
    
    # Check if model exists
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise Exception(
            f"Model not found at {LOCAL_MODEL_PATH}. "
            "Please run 'python download_model.py' first."
        )
    
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_MODEL_PATH)
    
    # Check if FAISS index exists and load
    if not os.path.exists(FAISS_INDEX_PATH):
        raise Exception(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Please run 'python build_index.py' first. "
            "Make sure you have text files in Raw-data-from-TTT/ directory."
        )
        
    try:
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings,
            allow_dangerous_deserialization=True  # Add this parameter
        )
    except Exception as e:
        raise Exception(
            f"Error loading FAISS index from {FAISS_INDEX_PATH}: {str(e)}\n"
            "The index may be corrupted. Try deleting it and running build_index.py again."
        ) from e
    
    # Create base chain without memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=True,
        return_generated_question=False,
    )
    
    # Wrap with message history
    qa_with_history = RunnableWithMessageHistory(
        qa_chain,
        lambda session_id: get_message_history(session_id),
        input_messages_key="question",
        history_messages_key="chat_history"
    )
    
    _qa_chain_dict[session_id] = qa_with_history
    return qa_with_history

def clear_memory(session_id: str):
    """Clear conversation memory for a specific session"""
    if session_id in _message_histories:
        _message_histories[session_id] = ChatMessageHistory()
    if session_id in _qa_chain_dict:
        del _qa_chain_dict[session_id]