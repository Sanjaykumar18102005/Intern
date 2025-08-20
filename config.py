import os

# Model choices
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# Vectorstore (local or cloud)
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index")
