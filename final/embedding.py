from sentence_transformers import SentenceTransformer
import numpy as np

def init_embedder(device="cpu"):
    # Load the BAAI bge-base-en-v1.5 model
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
    return model

def embed_texts(embedder, texts):
    # Important: BGE models need "passage: " prefix for passages
    texts = [f"passage: {text}" for text in texts]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize_embeddings(embeddings)
    return embeddings

def embed_query(embedder, query):
    # Important: BGE models need "query: " prefix for queries
    query = [f"query: {query}"]
    emb = embedder.encode(query, convert_to_numpy=True)
    emb = normalize_embeddings(emb)
    return emb

def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
