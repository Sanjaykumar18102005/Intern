from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.vectorstore import build_vectorstore, load_vectorstore
from utils.qa_pipeline import build_qa_pipeline
import shutil
import os

app = FastAPI()

VECTORSTORE_PATH = "vectorstore"

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    temp_file = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)

    build_vectorstore(temp_file, VECTORSTORE_PATH)
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
