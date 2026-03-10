# Semantic Search Engine

A lightweight, efficient semantic search engine built with Python that uses sentence embeddings to find contextually relevant documents based on meaning rather than keyword matching.

## Overview

This project implements a semantic search system using the Sentence-BERT (SBERT) model to convert text documents into dense vector embeddings. Unlike traditional keyword-based search, semantic search understands the contextual meaning of queries, enabling it to find relevant documents even when they don't share exact keywords.

## Features

- **Semantic Understanding**: Finds documents based on meaning and context, not just keywords
- **Fast Similarity Search**: Uses cosine similarity with NumPy for efficient vector comparisons
- **Pre-trained Models**: Leverages the `all-MiniLM-L6-v2` model for high-quality embeddings
- **Simple Architecture**: Clean separation between ingestion and search functionality
- **Extensible**: Easy to add new documents and scale to larger corpora

## Architecture

```
semantic-search-engine/
├── docs/                    # Source documents for indexing
│   ├── Ai.txt
│   ├── neural_networks.txt
│   ├── nlp.txt
│   ├── computer_vision.txt
│   ├── reinforcement_learning.txt
│   └── machinelearning.txt
├── data/                    # Generated embeddings and metadata
│   ├── embeddings.npy       # NumPy array of document embeddings
│   └── documents.json       # Document metadata and content
├── ingest.py               # Document ingestion and embedding generation
├── search.py               # Search interface and query processing
└── requirements.txt        # Python dependencies
```

## How It Works

### 1. Document Ingestion (ingest.py)

The ingestion pipeline processes documents through three stages:

1. **Document Loading**: Reads all `.txt` files from the `docs/` directory
2. **Embedding Generation**: Converts each document into a 384-dimensional vector using Sentence-BERT
3. **Persistence**: Saves embeddings as NumPy arrays and document metadata as JSON

### 2. Semantic Search (search.py)

The search process follows these steps:

1. **Query Encoding**: Converts the user's query into an embedding vector
2. **Similarity Calculation**: Computes cosine similarity between query and all document embeddings
3. **Ranking**: Sorts documents by similarity score and returns top-k results

### Mathematical Foundation

Cosine similarity is calculated as:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where:
- `A` is the query embedding
- `B` is a document embedding
- `·` represents dot product
- `||·||` represents vector magnitude

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd semantic-search-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies are:
- `sentence-transformers`: For generating semantic embeddings
- `torch`: PyTorch backend for neural network operations
- `numpy`: Efficient numerical operations

## Usage

### Step 1: Ingest Documents

Process your documents and generate embeddings:

```bash
python ingest.py
```

**Output:**
```
Documents loaded: 6
Batches: 100%|████████████| 1/1 [00:00<00:00, 2.34it/s]
Embeddings shape: (6, 384)
Embeddings and document metadata saved to data/
```

### Step 2: Search

Run the interactive search interface:

```bash
python search.py
```

**Example Session:**
```
Enter search query (or 'quit' to exit): image recognition

Top 3 results:

1. computer_vision.txt (Score: 0.7234)
   Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world...

2. Ai.txt (Score: 0.6891)
   Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers...

3. neural_networks.txt (Score: 0.6543)
   Neural networks are computational models inspired by the biological neural networks in animal brains...
```

## Adding New Documents

1. Add your `.txt` files to the `docs/` directory
2. Re-run the ingestion pipeline:
```bash
python ingest.py
```
3. The search engine will automatically index the new documents

## Performance Considerations

### Embedding Model

The `all-MiniLM-L6-v2` model offers an excellent balance:
- **Size**: 80MB
- **Dimensions**: 384
- **Speed**: ~14,000 sentences/second on CPU
- **Quality**: High semantic accuracy for general-purpose text

### Scalability

For larger document collections:
- **< 10,000 documents**: Current implementation works well
- **10,000 - 100,000 documents**: Consider using FAISS for approximate nearest neighbor search
- **> 100,000 documents**: Implement vector databases like Pinecone, Weaviate, or Milvus

### Memory Usage

- Each document embedding: 384 floats × 4 bytes = 1.5 KB
- 10,000 documents ≈ 15 MB of embeddings
- Document text stored separately in JSON

## Technical Details

### Sentence-BERT Model

The `all-MiniLM-L6-v2` model is:
- Fine-tuned on 1 billion sentence pairs
- Optimized for semantic similarity tasks
- Based on Microsoft's MiniLM architecture
- Trained using contrastive learning

### Similarity Scoring

Scores range from -1 to 1:
- **0.8 - 1.0**: Highly similar (near-duplicate content)
- **0.6 - 0.8**: Semantically related
- **0.4 - 0.6**: Somewhat related
- **< 0.4**: Low relevance

## Customization

### Change the Embedding Model

Edit `ingest.py` and `search.py`:

```python
# For multilingual support
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# For higher quality (slower)
model = SentenceTransformer("all-mpnet-base-v2")

# For faster inference (lower quality)
model = SentenceTransformer("all-MiniLM-L12-v2")
```

### Adjust Number of Results

Modify the `top_k` parameter in `search.py`:

```python
results = search(query, top_k=5)  # Return top 5 results
```

### Add Metadata Filtering

Extend the search function to filter by document attributes:

```python
def search(query, top_k=3, category=None):
    # Add filtering logic based on document metadata
    pass
```

## Limitations

- **Context Window**: Limited to 256 tokens per document (longer documents are truncated)
- **Language**: Optimized for English (use multilingual models for other languages)
- **Real-time Updates**: Requires re-running ingestion to add new documents
- **Exact Matches**: May miss exact keyword matches if semantically dissimilar

## Future Enhancements

- [ ] Add support for PDF and DOCX documents
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Add web interface with Flask/FastAPI
- [ ] Integrate vector database for production scalability
- [ ] Add document chunking for long texts
- [ ] Implement query expansion and re-ranking
- [ ] Add support for multi-modal search (text + images)

## Contributing

Contributions are welcome! Areas for improvement:
- Performance optimizations
- Additional document format support
- Enhanced search algorithms
- Better visualization of results

## License

MIT License - feel free to use this project for learning or commercial purposes.

## References

- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Semantic Search Guide](https://www.sbert.net/examples/applications/semantic-search/README.html)

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using Sentence Transformers and PyTorch**
