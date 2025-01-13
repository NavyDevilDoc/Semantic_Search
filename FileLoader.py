import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from docx import Document
from tkinter import Tk, filedialog
import logging

logging.basicConfig(level=logging.INFO)

class FileLoader:
    def __init__(self):
        pass

    def load_data(self, file_path):
        """
        Load data from a PDF, CSV, or DOCX file.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: Text content of the file.
        """
        try:
            if file_path.endswith('.pdf'):
                return self._load_pdf(file_path)
            elif file_path.endswith('.csv'):
                return self._load_csv(file_path)
            elif file_path.endswith('.docx'):
                return self._load_docx(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide a PDF, CSV, or DOCX file.")
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            return ""

    def _load_pdf(self, file_path):
        doc = fitz.open(file_path)
        text = ''
        for page in doc:
            text += page.get_text()
            if not text.strip():
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        return text

    def _load_csv(self, file_path):
        df = pd.read_csv(file_path)
        return df.to_string()

    def _load_docx(self, file_path):
        doc = Document(file_path)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text

    def select_files(self):
        """
        Open a file dialog to select PDF, CSV, or DOCX files and load their content.

        Returns:
            list: List of document contents.
        """
        root = Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(title="Select PDF, CSV, or DOCX files", filetypes=[("PDF files", "*.pdf"), ("CSV files", "*.csv"), ("Word files", "*.docx")])
        documents = []
        for file_path in file_paths:
            content = self.load_data(file_path)
            documents.append(content)
        return documents