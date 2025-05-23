from sentence_transformers import SentenceTransformer
import os

def download_and_save_model():
    print("Downloading model BAAI/bge-m3...")
    model = SentenceTransformer("BAAI/bge-m3")
    
    # Create directory if not exists
    os.makedirs("local_models", exist_ok=True)
    
    print("Saving model to local_models/bge-m3...")
    model.save("local_models/bge-m3")
    print("Model saved successfully!")

if __name__ == "__main__":
    download_and_save_model()
