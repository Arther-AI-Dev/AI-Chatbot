import requests
from typing import Optional, List, Dict, Any
from langchain.llms.base import LLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import glob

# ----------------------------------------
# 1. Define LocalLLM with pydantic fields
# ----------------------------------------
class LocalLLM(LLM):
    api_url: str
    model_name: str

    @property
    def _llm_type(self) -> str:
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
    


# ----------------------------------------
# 2. Instantiate LLM
# ----------------------------------------
API_URL = "http://localhost:5433"
MODEL_NAME = "scb10x/typhoon2.1-gemma3-4b:latest"
llm = LocalLLM(api_url=API_URL, model_name=MODEL_NAME)




# ----------------------------------------
# 3. Prepare embeddings
# ----------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")





# ----------------------------------------
# 4. Load documents from .txt files
# ----------------------------------------
paths = glob.glob("Raw-data-from-TTT\**\*.txt", recursive=True)
documents = []
for p in paths:
    loader = TextLoader(p, encoding="utf-8")
    documents.extend(loader.load())




# ----------------------------------------
# 5. Build or load FAISS index
# ----------------------------------------
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")




# ----------------------------------------
# 6. Define a PromptTemplate including chat history
# ----------------------------------------
custom_template = """\
บทสนทนาที่ผ่านมา:
{chat_history}

ใช้บริบทต่อไปนี้ในการตอบคำถามที่อยู่ท้ายบท
ผมคือ TTT-Assistant ผู้ช่วย AI ของบริษัท TTT Brothers Co., Ltd.
ตอบเป็นภาษาไทย หากคุณไม่ทราบคำตอบ ให้ตอบเพียงว่าคุณไม่ทราบ อย่าพยายามแต่งคำตอบขึ้นมา

ข้อมูลที่มี:
{context}

คำถาม: {question}

คำตอบ:"""

CUSTOM_PROMPT = PromptTemplate(
    template=custom_template,
    input_variables=["chat_history", "context", "question"]
)





# ----------------------------------------
# 7. Create memory for conversational history
# ----------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)





# ----------------------------------------
# 8. Build the Conversational Retrieval QA chain
# ----------------------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
)





# ----------------------------------------
# 9. Interactive chat loop
# ----------------------------------------
print("\nTTT Assistant พร้อมให้บริการ! (พิมพ์ 'exit()' เพื่อออกจากโปรแกรม)")

while True:
    try:
        question = input("\nคำถาม: ").strip()
        if question.lower() == 'exit()':
            print("ขอบคุณที่ใช้บริการ!")
            break
        if not question:
            continue

        result = qa_chain({"question": question})
        print("\nคำตอบ:", result["answer"])

    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {e}")