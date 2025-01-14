import chromadb

class DatabaseManager:
    def __init__(self, db_name):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(db_name)

    def add_documents(self, document_embeddings):
        """
        Add document embeddings to the collection.

        Args:
            document_embeddings (list of np.array): List of document embeddings.
        """
        ids = [str(i) for i in range(len(document_embeddings))]
        self.collection.add(ids=ids, embeddings=document_embeddings)

    def query_documents(self, query_embedding, top_k=5):
        results = self.collection.query(query_embedding, top_k=top_k)
        return results['ids']

    def get_all_documents(self):
        """
        Retrieve all documents from the collection.

        Returns:
            list of dict: List of documents with their embeddings.
        """
        return self.collection.get()

    def get_all_embeddings(self):
        """
        Retrieve all document embeddings from the collection.

        Returns:
            list of np.array: List of document embeddings.
        """
        documents = self.get_all_documents()
        return [doc['embedding'] for doc in documents]