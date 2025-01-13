import logging
from FileLoader import FileLoader
from TextPreprocessor import TextPreprocessor
from EmbeddingGenerator import EmbeddingGenerator
from DatabaseManager import DatabaseManager

logging.basicConfig(level=logging.INFO)

class SemanticClustering:
    def __init__(self):
        self.file_loader = FileLoader()
        self.text_preprocessor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator()
        self.database_manager = DatabaseManager()

    def run(self, query, top_k):
        try:
            documents = self.file_loader.select_files()
            if not documents:
                raise ValueError("No documents loaded. Please select valid PDF, CSV, or DOCX files.")
            logging.info("Documents loaded successfully.")
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            exit(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred during document selection: {e}")
            exit(1)

        try:
            preprocessed_documents = [self.text_preprocessor.preprocess_text(doc) for doc in documents]
            document_embeddings = self.embedding_generator.generate_embeddings(preprocessed_documents)
        except Exception as e:
            logging.error(f"An error occurred during preprocessing or embedding generation: {e}")
            exit(1)

        try:
            self.database_manager.add_documents(document_embeddings)
        except Exception as e:
            logging.error(f"An error occurred during document addition to the database: {e}")
            exit(1)

        try:
            preprocessed_query = self.text_preprocessor.preprocess_text(query)
            query_embedding = self.embedding_generator.model.encode([preprocessed_query])[0]
            top_doc_ids = self.database_manager.query_documents(query_embedding, top_k)
            top_docs = [documents[int(doc_id)] for doc_id in top_doc_ids]
            for i, doc in enumerate(top_docs):
                logging.info(f"Document {i+1}: {doc}")
        except Exception as e:
            logging.error(f"An error occurred during query processing or output: {e}")
            exit(1)