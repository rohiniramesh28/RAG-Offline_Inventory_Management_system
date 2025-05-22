from embedding import init_embedder, embed_texts, embed_query
from faiss_index import build_index, query_index
from llm_inference import load_llm, generate_answer

SIMILARITY_THRESHOLD = 0.4
SIMPLIFY_INSTRUCTION = "Please provide a clear, concise, and simple answer without extra explanations."

class QASystem:
    def __init__(self, texts, device="cpu"):
        self.embedder = init_embedder(device)
        self.texts = texts
        self.embeddings = embed_texts(self.embedder, texts)
        self.index = build_index(self.embeddings)
        self.tokenizer, self.model = load_llm(device=device)
        self.device = device

    def answer(self, question, top_k=3):
        q_emb = embed_query(self.embedder, question)
        distances, indices = query_index(self.index, q_emb, top_k)
        max_sim = float(distances[0][0])

        if max_sim >= SIMILARITY_THRESHOLD:
            context = "\n".join([self.texts[i] for i in indices[0]])
            prompt = (
                f"Context:\n{context}\n\n"
                f"{SIMPLIFY_INSTRUCTION}\n"
                f"Answer the question based only on the context.\n"
                f"Question: {question}\nAnswer:"
            )
            answer = generate_answer(self.tokenizer, self.model, prompt, self.device)
        else:
            prompt = (
                f"{SIMPLIFY_INSTRUCTION}\n"
                f"Question: {question}\nAnswer:"
            )
            answer = generate_answer(self.tokenizer, self.model, prompt, self.device)

        return answer
