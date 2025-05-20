# Import standard libraries
import os  
import re 
import string 
import uuid  
import shutil  
import uvicorn  
from typing import List 

# Import FastAPI components
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  # (Optional) To serve static files like CSS/JS
from pydantic import BaseModel 

# OCR and file handling libraries
import pytesseract
from PIL import Image
import docx
import pandas as pd
import fitz

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Vector search and embedding libraries
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Google Gemini setup
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from file
load_dotenv("API.env")
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
if not api_key:
    raise Exception("GOOGLE_API_KEY environment variable not found!")
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# Initialize FastAPI app
app = FastAPI()

# Setup Jinja2 template directory
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")  # Optional: serve static files

# Create upload folder if it doesn't exist
UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".docx", ".txt", ".csv", ".xlsx"}

# Download NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))  # Set of stopwords
lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
metadata = []  # Stores filenames
doc_texts = []  # Stores cleaned text for each document

# Helper function to get file extension
def get_extension(filename: str):
    return os.path.splitext(filename)[1].lower()

# Remove unwanted text from PDF pages
def remove_page_artifacts(text: str) -> str:
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        if re.match(r"^\s*\d+\s*$", line):  # Skip page numbers
            continue
        if "doi" in line.lower() or "www." in line.lower():  # Skip links/DOIs
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

# Fix broken paragraph issues from extracted text
def fix_broken_paragraphs(text: str) -> str:
    lines = text.split("\n")
    fixed_text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line and not line.endswith((".", "?", "!", ":")):
            fixed_text += line + " "
        else:
            fixed_text += line + "\n"
    return fixed_text

# Clean and preprocess raw text
def preprocess_text(text: str) -> str:
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = text.encode("ascii", errors="ignore").decode()  # Remove non-ascii

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords

    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    return " ".join(unique_tokens)

# Extract raw text from different file types
def extract_text_from_file(file_path: str, ext: str) -> str:
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".docx":
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)

        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return df.to_string()

        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            return df.to_string()

        elif ext in {".jpg", ".jpeg", ".png"}:
            pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)

        elif ext == ".pdf":
            text = ""
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("text") + "\n"
            text = remove_page_artifacts(text)
            text = fix_broken_paragraphs(text)
            return text

        else:
            return ""
    except Exception as e:
        return f"Error extracting content: {str(e)}"

# Home route - renders index.html
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Upload and process documents
@app.post("/upload-read-preprocess/", response_class=HTMLResponse)
async def upload_read_preprocess(request: Request, files: List[UploadFile] = File(...)):
    if len(files)<75: # for test purpose we can make it 2-3
        raise HTTPException(status_code=400, detail="less files. Min 75 required.")

    processed_data = {}

    for file in files:
        ext = get_extension(file.filename)
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {ext}")

        unique_name = f"{uuid.uuid4().hex}{ext}"  # Unique filename
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)  # Save file to disk

        raw_text = extract_text_from_file(file_path, ext)  # Extract text
        cleaned_text = preprocess_text(raw_text)  # Preprocess it
        processed_data[file.filename] = cleaned_text  # Store result as key:value pair in dictonary(key=file name ans value=data)

        embedding = embedding_model.encode(cleaned_text, convert_to_numpy=True).astype('float32')  # Get embedding
        index.add(embedding.reshape(1, -1))  # Add to FAISS index
        metadata.append(file.filename)
        doc_texts.append(cleaned_text)  

    return JSONResponse(content=processed_data)

# Pydantic model for search query
class QueryRequest(BaseModel):
    query_text: str  # User query
    top_k: int = 10  # Number of results to return

# Semantic search endpoint
@app.post("/search/")
async def search(query: QueryRequest):
    query_vec = embedding_model.encode(query.query_text)  # Encode query
    D, I = index.search(np.array([query_vec], dtype="float32"), query.top_k)  # Search top_k nearest

    results = []
    context = ""

    for idx, dist in zip(I[0], D[0]):
        if idx < len(metadata):
            results.append({"filename": metadata[idx], "distance": float(dist)})
            context += doc_texts[idx] + " "  # Collect text for response generation

    context = context[:2000]  # Limit context size

    prompt = f"""
    Answer the question based on the context below.
    Context: {context}
    Question: {query.query_text}
    Answer:
    """

    try:
        response = gemini_model.generate_content(prompt)  # Get Gemini response
        answer = response.text.strip() if response.text else "No answer generated."  # Extract response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error using Gemini: {str(e)}")

    return {
        "query": query.query_text,
        "answer": answer,
        "results": results
    }

# Run FastAPI app if this is the main script
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Read port from env or default 8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)  # Start server
