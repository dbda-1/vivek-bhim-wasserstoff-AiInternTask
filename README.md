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

## 📁 Folder Structure
  project/
    ├── main.py # Main FastAPI application
    ├── templates/ # HTML templates (e.g., index.html)
    ├── uploaded_documents/ # Folder to store uploaded files
    ├── API.env # Google API key file
    ├── README.md # Project documentation
    └── requirements.txt # Python dependencies

