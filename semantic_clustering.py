import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import contractions
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from docx import Document
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
                # Perform OCR if text extraction fails
                if not text.strip():
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img)
            return text
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return df.to_string()
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            text = ''
            for paragraph in doc.paragraphs:
                text += paragraph.text + '\n'
            return text
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
    file_paths = filedialog.askopenfilenames(title="Select PDF, CSV, or DOCX files", filetypes=[("PDF files", "*.pdf"), ("CSV files", "*.csv"), ("Word files", "*.docx")])
    documents = []
    for file_path in file_paths:
        content = load_data(file_path)
        documents.append(content)
    return documents


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('ibm-granite/granite-embedding-30m-english')
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# Preprocess text by tokenizing and removing stopwords
def preprocess_text(text):
    """
    Preprocess text by tokenizing and removing stopwords.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    # Expand contractions
    text = contractions.fix(text)
    # Lowercase the text
    text = text.lower()
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    # Spell check and correct tokens
    tokens = [spell.correction(token) for token in tokens]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Remove extra whitespace
    text = ' '.join(tokens)
    return text


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
    model = SentenceTransformer('ibm-granite/granite-embedding-30m-english')
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