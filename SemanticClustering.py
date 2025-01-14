import logging
from FileLoader import FileLoader
from TextPreprocessor import TextPreprocessor
from EmbeddingGenerator import EmbeddingGenerator
from DatabaseManager import DatabaseManager
from Scoring import Scoring

logging.basicConfig(level=logging.INFO)

class SemanticClustering:
    def __init__(self, db_name):
        self.file_loader = FileLoader()
        self.text_preprocessor = TextPreprocessor()
        self.embedding_generator = EmbeddingGenerator()
        self.database_manager = DatabaseManager(db_name)
        self.scoring = Scoring()

    def load_data(self):
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

    def query_data(self, query, top_k):
        try:
            preprocessed_query = self.text_preprocessor.preprocess_text(query)
            query_embedding = self.embedding_generator.model.encode([preprocessed_query])[0]
            document_embeddings = self.database_manager.get_all_embeddings()
            scored_documents = self.scoring.compute_scores(query_embedding, document_embeddings)
            top_scored_documents = scored_documents[:top_k]
            documents = self.file_loader.select_files()  # Load documents again to retrieve content
            for i, (doc_id, score) in enumerate(top_scored_documents):
                logging.info(f"Document {i+1} (Score: {score}): {documents[doc_id]}")
        except Exception as e:
            logging.error(f"An error occurred during query processing or output: {e}")
            exit(1)