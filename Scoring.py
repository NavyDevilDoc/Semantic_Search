import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Scoring:
    def __init__(self):
        pass

    def compute_scores(self, query_embedding, document_embeddings):
        """
        Compute cosine similarity scores between the query embedding and document embeddings.

        Args:
            query_embedding (np.array): Embedding of the query.
            document_embeddings (list of np.array): List of document embeddings.

        Returns:
            list of tuple: List of (index, score) tuples sorted by score in descending order.
        """
        query_embedding = np.array(query_embedding).reshape(1, -1)
        document_embeddings = np.array(document_embeddings)
        scores = cosine_similarity(query_embedding, document_embeddings)[0]
        scored_documents = [(i, score) for i, score in enumerate(scores)]
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents