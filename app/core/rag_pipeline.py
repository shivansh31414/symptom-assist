"""
rag_pipeline.py
---------------
RAG (Retrieval-Augmented Generation) pipeline for the symptom chatbot.

This module loads curated medical documents from data/medical_docs.csv,
builds dense embeddings with sentence-transformers, and uses cosine similarity
to retrieve the most relevant context for the LLM prompt.

Flow:
    1. Load medical documents from CSV at startup
    2. Embed each document with a sentence transformer model
    3. Embed the user query at retrieval time
    4. Score documents with cosine similarity in embedding space
    5. Return the best matches as context for the LLM prompt
"""

import csv
import os

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - dependency issue is environment-specific
    raise ImportError(
    "sentence-transformers is required for semantic retrieval. "
    "Install dependencies with pip install -r requirements.txt."
    ) from exc


# ---------------------------------------------------------------------------
# 1. CSV Loader
# ---------------------------------------------------------------------------

def load_documents_from_csv(csv_path: str) -> list[dict]:
    """
    Read medical_docs.csv and return a list of document dicts:
      {id, condition, title, content}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Medical docs CSV not found: {csv_path}")

    documents: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documents.append({
                "id":        f"doc_{row['condition'].strip()}",
                "condition": row["condition"].strip(),
                "title":     row["title"].strip(),
                "content":   row["content"].strip(),
            })

    print(f"[RAG] Loaded {len(documents)} medical documents from CSV")
    return documents


# ---------------------------------------------------------------------------
# 2. Semantic embedding retriever
# ---------------------------------------------------------------------------

class SemanticEmbeddingRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents: list[dict] = []
        self.document_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def _document_text(self, document: dict) -> str:
        return " ".join(
            part
            for part in [
                document.get("title", ""),
                document.get("condition", ""),
                document.get("content", ""),
            ]
            if part
        ).strip()

    def index(self, documents: list[dict]):
        self.documents = documents

        if not documents:
            self.document_embeddings = np.empty((0, 0), dtype=np.float32)
            print("[RAG] No medical documents found; semantic index is empty.")
            return

        corpus = [self._document_text(document) for document in documents]
        embeddings = self.model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.document_embeddings = np.asarray(embeddings, dtype=np.float32)

        print(
            f"[RAG] Indexed {len(documents)} documents using "
            f"{self.model_name} embeddings (dim={self.document_embeddings.shape[1]})"
        )

    def embed_query(self, query: str) -> np.ndarray:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(query_embedding[0], dtype=np.float32)

    def _retrieve_from_vector(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        if not self.documents or self.document_embeddings.size == 0:
            return []

        normalized_query_vector = np.asarray(query_vector, dtype=np.float32)
        scores = self.document_embeddings @ normalized_query_vector
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            document = self.documents[int(index)].copy()
            document["relevance_score"] = round(score, 4)
            results.append(document)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        query_vector: np.ndarray | None = None,
    ) -> list[dict]:
        if not self.documents or self.document_embeddings.size == 0:
            return []

        if query_vector is None:
            query_vector = self.embed_query(query)

        return self._retrieve_from_vector(query_vector, top_k=top_k)


# ---------------------------------------------------------------------------
# 3. RAG Pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, csv_path: str | None = None):
        if csv_path and os.path.exists(csv_path):
            documents = load_documents_from_csv(csv_path)
        else:
            # Fallback: empty (should not normally reach here)
            documents = []
            print("[RAG] WARNING: no medical_docs.csv found; RAG context will be empty.")

        self.retriever = SemanticEmbeddingRetriever()
        self.retriever.index(documents)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Return formatted context string for the LLM prompt."""
        docs = self.retriever.retrieve(query, top_k=top_k)
        if not docs:
            return ""
        parts = [f"[{doc['title']}]\n{doc['content']}" for doc in docs]
        return "\n\n---\n\n".join(parts)

    def retrieve_raw(self, query: str, top_k: int = 3) -> list[dict]:
        """Return raw document list with scores — useful for debugging."""
        return self.retriever.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# 4. Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    _here = pathlib.Path(__file__).parent.parent.parent
    csv_p = str(_here / "data" / "medical_docs.csv")

    rag = RAGPipeline(csv_path=csv_p)
    for q in [
        "I have a headache that throbs on one side with light sensitivity",
        "burning when I urinate and need to go frequently",
        "stomach cramps and vomiting after eating out",
    ]:
        print(f"\nQuery: '{q}'")
        for r in rag.retrieve_raw(q, top_k=2):
            print(f"  → {r['title']} (score: {r['relevance_score']})")
