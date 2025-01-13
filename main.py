from SemanticClustering import SemanticClustering

if __name__ == "__main__":
    query = input("Enter your search query: ")
    top_k = int(input("Enter the number of top documents to retrieve: "))
    clustering = SemanticClustering()
    clustering.run(query, top_k)