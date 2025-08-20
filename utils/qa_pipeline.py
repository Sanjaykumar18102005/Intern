from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_MODEL

def build_qa_pipeline(vectorstore):
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa
