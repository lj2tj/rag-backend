# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from api import upload_document, get_files, embed_document, ask_question, AskRequest

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://localhost:3000",
    "http://127.0.0.1:3000",
    "https://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_document")
async def upload_document_endpoint(
    file: UploadFile = File(...),
):
    return await upload_document(file)


@app.get("/files")
def get_files_endpoint():
    return get_files()


@app.post("/embed_document")
async def embed_document_endpoint(
    file_path: str = Form(...)
):
    return await embed_document(file_path)


@app.post("/ask")
async def ask_question_endpoint(request: AskRequest):
    return await ask_question(request)


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=3001, 
        workers=4,
        timeout_keep_alive=900)
