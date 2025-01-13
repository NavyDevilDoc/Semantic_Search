import chromadb

class DatabaseManager:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")

    def add_documents(self, document_embeddings):
        for i, embedding in enumerate(document_embeddings):
            self.collection.add_document(str(i), embedding)

    def query_documents(self, query_embedding, top_k=5):
        results = self.collection.query(query_embedding, top_k=top_k)
        return results['ids']