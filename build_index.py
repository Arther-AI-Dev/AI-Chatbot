from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import glob

def build_faiss_index():
    print("Starting to build FAISS index...")    # Initialize embeddings with local model
    embeddings = HuggingFaceEmbeddings(model_name="local_models/bge-m3")
    
    # Load documents
    paths = glob.glob("Raw-data-from-TTT/**/*.txt", recursive=True)
    documents = []
    for p in paths:
        print(f"Loading document: {p}")
        loader = TextLoader(p, encoding="utf-8")
        documents.extend(loader.load())
    
    # Create and save vectorstore
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index built and saved successfully!")

if __name__ == "__main__":
    build_faiss_index()
