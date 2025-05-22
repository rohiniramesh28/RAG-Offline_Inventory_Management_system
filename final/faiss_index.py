import faiss
import numpy as np
import pickle
import os

def build_index(embeddings):
    embeddings = normalize_embeddings(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    return index

def query_index(index, query_embedding, top_k=3):
    query_embedding = normalize_embeddings(query_embedding)
    query_embedding = query_embedding.astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def save_index(index, doc_mapping, path="faiss_store"):
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "vector.index"))
    with open(os.path.join(path, "doc_mapping.pkl"), "wb") as f:
        pickle.dump(doc_mapping, f)
    print(f"[✅] Index and mapping saved to '{path}/'")


def load_index(path="faiss_store"):
    index = faiss.read_index(os.path.join(path, "vector.index"))
    with open(os.path.join(path, "doc_mapping.pkl"), "rb") as f:
        doc_mapping = pickle.load(f)
    print(f"[✅] Index and mapping loaded from '{path}/'")
    return index, doc_mapping
