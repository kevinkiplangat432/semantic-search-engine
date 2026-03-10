import numpy as np
from sentence_transformers import SentenceTransformer
import json

model= SentenceTransformer("all-MiniLM-L6-v2")

embeddings = np.load('data/embeddings.npy')

with open("data/documents.json", 'r', encoding='utf-8') as f:
    documents= json.load(f)
 
def search(query, top_k):
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = np.dot( embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "filename": documents[idx]["filename"],
            "score": float(similarities[idx]),
            "content": documents[idx]["content"][:200] + "..."
        })
    return results


if __name__ == "__main__":
    while True:
        query = input("\nEnter search query (or 'quit' to exit): ") # the users input
        if query.lower() == 'quit':  # allow the user to quit
            break 
        top_k = input("\nEnter the maximum number of search result you want to see")
        results = search(query, top_k) 
        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']} (Score: {result['score']:.4f})")
            print(f"   {result['content']}")
    
    



   
