import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_MODEL, VECTORSTORE_PATH

def build_vectorstore(pdf_file, save_path=VECTORSTORE_PATH):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectorstore.save_local(save_path)
    return vectorstore

def load_vectorstore(path=VECTORSTORE_PATH):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore
