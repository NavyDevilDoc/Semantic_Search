from SemanticClustering import SemanticClustering

if __name__ == "__main__":
    action = input("Enter 'load' to load data or 'query' to perform a query: ").strip().lower()
    db_name = input("Enter the database name: ").strip()
    
    clustering = SemanticClustering(db_name)
    
    if action == 'load':
        clustering.load_data()
    elif action == 'query':
        query = input("Enter your search query: ")
        top_k = int(input("Enter the number of top documents to retrieve: "))
        clustering.query_data(query, top_k)
    else:
        print("Invalid action. Please enter 'load' or 'query'.")