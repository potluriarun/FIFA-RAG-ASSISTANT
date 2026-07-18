"""Pipeline B step 1: question -> top-k relevant chunks from Chroma.

Run: python src/retrieve.py
Type a question, eyeball the retrieved chunks. Blank line / Ctrl+C to quit.
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "fifa_rules"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # must match the model used in ingest.py
TOP_K = 5

_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Embed `question` with the same model used at ingest time and return
    the top_k nearest chunks, each with its source/page/law metadata."""
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": text,
            "source": meta.get("source", ""),
            "page": meta.get("page", ""),
            "law": meta.get("law", ""),
            "distance": distance,
        })
    return hits


def format_hit(hit: dict, index: int, preview_chars: int = 300) -> str:
    label = f"{hit['source']} p.{hit['page']}"
    if hit["law"]:
        label += f" | {hit['law']}"
    preview = hit["text"][:preview_chars].replace("\n", " ")
    return f"[{index}] ({hit['distance']:.3f}) {label}\n    {preview}..."


def main():
    print(f"Loaded collection '{COLLECTION_NAME}' from {CHROMA_DIR}")
    print("Type a question (blank line to quit):\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        hits = retrieve(question)
        for i, hit in enumerate(hits, start=1):
            print(format_hit(hit, i))
        print()


if __name__ == "__main__":
    main()
