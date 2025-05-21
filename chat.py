import requests
from typing import Optional, List, Dict, Any
from langchain.llms.base import LLM

class LocalLLM(LLM):
    # ประกาศฟิลด์ให้ Pydantic รู้จัก
    api_url: str
    model_name: str

    @property
    def _llm_type(self) -> str:
        """ชื่อชนิดของ LLM เพื่อ logging ภายใน LangChain"""
        return "local_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model_name, "api_url": self.api_url}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """เรียก API และคืน response มาเป็นสตริง"""
        res = requests.post(
            f"{self.api_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            },
        )
        res.raise_for_status()
        return res.json().get("response", "")

API_URL = "http://localhost:5433"
MODEL_NAME = "scb10x/typhoon2.1-gemma3-4b:latest"
llm = LocalLLM(api_url=API_URL, model_name=MODEL_NAME)


from langchain.embeddings import HuggingFaceEmbeddings
# Wrap SentenceTransformer ผ่าน LangChain
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

from langchain.document_loaders import TextLoader
import glob

paths = glob.glob("Raw-data-from-TTT\**\*.txt", recursive=True)
documents = []
for p in paths:
    loader = TextLoader(p, encoding="utf-8")
    documents.extend(loader.load())


from langchain.vectorstores import FAISS
# สร้างและเก็บ Document embeddings ลงใน FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)
# บันทึก index ไว้ใช้งานครั้งถัดไป
vectorstore.save_local("faiss_index")

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Custom prompt template
custom_template = """ใช้บริบทต่อไปนี้ในการตอบคำถามที่อยู่ท้ายบท
ผมคือ TTT-Assistant ผู้ช่วย AI ของบริษัท TTT Brothers Co., Ltd.
หากคุณไม่ทราบคำตอบ ให้ตอบเพียงว่าคุณไม่ทราบ อย่าพยายามแต่งคำตอบขึ้นมา

ข้อมูลที่มี:
{context}

คำถาม: {question}

คำตอบ: """

CUSTOM_PROMPT = PromptTemplate(
    template=custom_template,
    input_variables=["context", "question"]
)

# สร้าง Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# สร้าง QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
)

# แทนที่โค้ดตัวอย่างด้วย interactive chat loop
print("\nTTT Assistant พร้อมให้บริการ! (พิมพ์ 'exit' เพื่อออกจากโปรแกรม)")

while True:
    try:
        question = input("\nคำถาม: ").strip()
        
        if question.lower() == 'exit':
            print("ขอบคุณที่ใช้บริการ!")
            break
            
        if not question:
            continue
            
        result = qa_chain({"question": question})
        print("\nคำตอบ:", result["answer"])
        
    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {str(e)}")