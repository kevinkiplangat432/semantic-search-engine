import os
import numpy as np
from sentence_transformers import SentenceTransformer
import json


DOCS_PATH = "docs"
DATA_PATH = "data"

model= SentenceTransformer("all-MiniLM-L6-v2")


def load_docs():
   documents = []
   filenames = []
   
   for filename in os.listdir(DOCS_PATH):
       if filename.endswith(".txt"):
              with open(os.path.join(DOCS_PATH, filename), "r", encoding="utf-8") as f:
                  content = f.read()
                  
                  documents.append(content)
                  filenames.append(filename)
   return documents, filenames


def generate_embeddings(documents, model):
    # Generate embeddings for all documents using the provided model
    # each document is converted into a vector representation (embedding) that captures its semantic meaning
    # 384 `dimensions for the "all-MiniLM-L6-v2" model
    embeddings = model.encode(
        documents, 
        show_progress_bar=True,
        convert_to_numpy=True)
    return embeddings


def save_embeddings_and_docs(embeddings, documents, filename):
    data_to_save = []
    for i in range(len(documents)):
        data_to_save.append({
            "filename": filename[i],
            "content": documents[i]
        })
    np.save(os.path.join(DATA_PATH, "embeddings.npy"), embeddings)
    
    with open(os.path.join(DATA_PATH, "documents.json"), "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2)

if __name__ == "__main__":
    documents, filenames = load_docs()
    print("Documents loaded:", len(documents))
    
    embeddings = generate_embeddings(documents,model)
    print("Embeddings shape:", embeddings.shape)
    
    save_embeddings_and_docs(embeddings, documents, filenames)
    print("Embeddings and document metadata saved to data/") 