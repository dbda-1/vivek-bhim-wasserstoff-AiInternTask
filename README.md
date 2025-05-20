# 📘 Document Research & Theme Identification Chatbot

This is a **FastAPI-based application** that allows users to upload various document types (PDF, DOCX, TXT, CSV, XLSX, images), extract and preprocess text, perform semantic search using FAISS, and get answers using **Google Gemini AI**.

---

## 🚀 Features

- 📄 Upload and process multiple documents  
- 🧠 Text extraction from PDF, DOCX, TXT, CSV, Excel, and images  
- 🔤 NLP preprocessing: lowercasing, punctuation removal, lemmatization, stopword removal  
- 🔍 Semantic search using Sentence Transformers and FAISS  
- 🤖 Answer generation via Google Gemini API  
- ⚙️ FastAPI-based web backend

---

## 🛠 Tech Stack

- **Framework**: FastAPI  
- **OCR**: Tesseract OCR, Pillow  
- **PDF/Image Processing**: PyMuPDF (fitz), pytesseract  
- **Text Preprocessing**: NLTK  
- **Embedding**: Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Vector Store**: FAISS  
- **LLM**: Google Gemini (`gemini-2.5-flash-preview-04-17`)

---

## 🔧 How to Run

  **1. Create and Activate Virtual Environment**  
    python -m venv venv    
  **2. Install Dependencies**  
    pip install -r requirements.txt  
  **3. Setup Google Gemini API**  
    Create a file named `.env` or `API.env`  
    Add your Gemini API key:  
    GOOGLE_API_KEY=your_google_gemini_api_key  
  **4. Run the Application**  
    python main.py

## 📖 Usage

After starting the application, open your browser to the provided URL (e.g., http://localhost:8000). You will see a user-friendly web interface where you can:

   1. Upload multiple documents in formats such as PDF, DOCX, TXT, CSV, Excel files, and images.

   2. View the extracted and preprocessed text from the uploaded files.
 
   3. Use the question input box at the bottom to ask queries related to the uploaded documents.

   4.  Receive context-aware answers powered by Google Gemini AI, based on semantic search over your documents



