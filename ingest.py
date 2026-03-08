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
    embeddings = model.encode(
        documents, 
        show_progress_bar=True,
        convert_to_numpy=True)
    return embeddings
    
    

if __name__ == "__main__":
    documents, filenames = load_docs()
    print("Documents loaded:", len(documents))
    
    embeddings = generate_embeddings(documents,model)
    print("Embeddings shape:", embeddings.shape)