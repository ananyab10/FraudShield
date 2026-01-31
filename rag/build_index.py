"""
Build a FAISS vector index from markdown knowledge files.

This script:
- Loads all `.md` files from `rag/knowledge/`
- Splits documents into character chunks (max 500 chars)
- Encodes chunks using sentence-transformers (all-MiniLM-L6-v2)
- Builds a FAISS index using cosine similarity (via L2-normalized vectors)
- Saves:
    - FAISS index to `rag/vector.index`
    - Text chunks to `rag/chunks.pkl`

No LLM calls are made here.
"""

import glob
import os
import pickle
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
VECTOR_INDEX_PATH = os.path.join(BASE_DIR, "vector.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHARS = 500


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_markdown_files(directory: str) -> List[str]:
    """Load all markdown files from the knowledge directory."""
    if not os.path.isdir(directory):
        return []

    paths = sorted(glob.glob(os.path.join(directory, "*.md")))
    documents: List[str] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            documents.append(f.read())

    return documents


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """Split text into chunks without cutting words abruptly."""
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)

        if end < length:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


def build_faiss_index(chunks: List[str]):
    """Create a FAISS index from text chunks."""
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Ensure float32
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)

    documents = load_markdown_files(KNOWLEDGE_DIR)

    all_chunks: List[str] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))

    if not all_chunks:
        raise RuntimeError(
            f"No markdown files found in {KNOWLEDGE_DIR}. "
            "Add .md files before building the index."
        )

    index = build_faiss_index(all_chunks)

    # Save index and chunks
    faiss.write_index(index, VECTOR_INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Indexed {len(all_chunks)} text chunks")
    print(f"📦 FAISS index saved to: {VECTOR_INDEX_PATH}")
    print(f"📄 Chunks saved to: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
