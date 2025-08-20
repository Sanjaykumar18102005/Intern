from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.vectorstore import build_vectorstore, load_vectorstore
from utils.qa_pipeline import build_qa_pipeline
import shutil
import os
import tempfile

app = FastAPI()

VECTORSTORE_PATH = "/tmp/vectorstore"  # Use /tmp for serverless writable storage

# Ensure vectorstore folder exists
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Use a secure temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    # Build vectorstore from uploaded PDF
    build_vectorstore(temp_file_path, VECTORSTORE_PATH)

    return {"message": "Vectorstore built successfully."}


@app.post("/ask/")
async def ask_question(query: str):
    try:
        vectorstore = load_vectorstore(VECTORSTORE_PATH)
        qa = build_qa_pipeline(vectorstore)
        response = qa.invoke({"query": query})["result"]
        return JSONResponse({"answer": response})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
