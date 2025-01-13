import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from tkinter import Tk, filedialog
import logging

logging.basicConfig(level=logging.INFO)


# Function to load data from PDF or CSV
def load_data(file_path):
    """
    Load data from a PDF or CSV file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Text content of the file.
    """
    try:
        if file_path.endswith('.pdf'):
            doc = fitz.open(file_path)
            text = ''
            for page in doc:
                text += page.get_text()
            return text
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return df.to_string()
        else:
            raise ValueError("Unsupported file format. Please provide a PDF or CSV file.")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return ""


# Data Ingestion using tkinter to select files
def select_files():
    """
    Open a file dialog to select PDF or CSV files and load their content.

    Returns:
        list: List of document contents.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(title="Select PDF or CSV files", filetypes=[("PDF files", "*.pdf"), ("CSV files", "*.csv")])
    documents = []
    for file_path in file_paths:
        content = load_data(file_path)
        if content:
            documents.append(content)
    return documents


# Preprocess text by tokenizing and removing stopwords
def preprocess_text(text):
    """
    Preprocess text by tokenizing and removing stopwords.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


try:
    # Select and load documents
    documents = select_files()
    if not documents:
        raise ValueError("No documents loaded. Please select valid PDF or CSV files.")
    logging.info("Documents loaded successfully.")
except ValueError as ve:
    print(f"ValueError: {ve}")
    logging.error(f"ValueError: {ve}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred during document selection: {e}")
    logging.error(f"An unexpected error occurred during document selection: {e}")
    exit(1)


try:
    # Preprocess documents
    preprocessed_documents = [preprocess_text(doc) for doc in documents]
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Get embeddings for documents
    batch_size = 32
    document_embeddings = []
    for i in range(0, len(preprocessed_documents), batch_size):
        batch_embeddings = model.encode(preprocessed_documents[i:i+batch_size])
        document_embeddings.extend(batch_embeddings)
except Exception as e:
    print(f"An error occurred during preprocessing or embedding generation: {e}")
    exit(1)


try:
    # Initialize ChromaDB
    client = chromadb.Client()
    collection = client.create_collection("documents")
    # Add documents to ChromaDB
    for i, embedding in enumerate(document_embeddings):
        collection.add_document(str(i), embedding)
except Exception as e:
    print(f"An error occurred during ChromaDB initialization or document addition: {e}")
    exit(1)


try:
    # Query Processing
    query = "Your search query"
    preprocessed_query = preprocess_text(query)
    query_embedding = model.encode([preprocessed_query])[0]
    # Matching
    results = collection.query(query_embedding, top_k=5)
    top_docs = [documents[int(doc_id)] for doc_id in results['ids']]
    # Output the top documents
    for i, doc in enumerate(top_docs):
        print(f"Document {i+1}:")
        print(doc)
        print("\n")
except Exception as e:
    print(f"An error occurred during query processing or output: {e}")
    logging.error(f"An error occurred during query processing or output: {e}")
    exit(1)