from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name='ibm-granite/granite-embedding-30m-english'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents, batch_size=32):
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch_embeddings = self.model.encode(documents[i:i+batch_size])
            embeddings.extend(batch_embeddings)
        return embeddings