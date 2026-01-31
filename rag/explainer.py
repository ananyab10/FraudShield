"""Retrieval-augmented explainer for fraud decision reason codes.

This module loads a FAISS index and associated text chunks, converts a
reason code to a natural-language query, retrieves the top-3 most
relevant knowledge chunks, scrubs possible PII, and returns a concise,
deterministic explanation that references behavior patterns (not users).

No external APIs are used; all work is local and deterministic.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(__file__)
VECTOR_INDEX_PATH = os.path.join(BASE_DIR, "vector.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


_model = None
_index = None
_chunks: List[str] | None = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_index_and_chunks() -> Tuple[faiss.Index, List[str]]:
    global _index, _chunks
    if _index is None:
        if not os.path.exists(VECTOR_INDEX_PATH):
            raise FileNotFoundError(f"Vector index not found at {VECTOR_INDEX_PATH}")
        _index = faiss.read_index(VECTOR_INDEX_PATH)
    if _chunks is None:
        if not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "rb") as fh:
            _chunks = pickle.load(fh)
    return _index, _chunks


def _code_to_query(reason_code: str) -> str:
    s = reason_code or ""
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lower()
    # produce a short natural-language query
    return f"guidance about {s}"


def _scrub_pii(text: str) -> str:
    # remove email-like tokens
    text = re.sub(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,6}\b", "[REDACTED]", text)
    # redact long digit sequences (accounts, phone numbers, UPIs, etc.)
    text = re.sub(r"\d{5,}", "[REDACTED]", text)
    # redact tokens that look like handles with @bank (UPI handles)
    text = re.sub(r"\b[\w.-]+@[\w.-]+\b", "[REDACTED]", text)
    return text


def _extract_relevant_sentences(chunk: str, query_tokens: List[str], max_sentences: int = 2) -> List[str]:
    # split on sentence-like boundaries
    parts = re.split(r"(?<=[.!?])\s+|\n+", chunk)
    selected: List[str] = []
    lowered_tokens = {t.lower() for t in query_tokens if t}
    for part in parts:
        if not part or len(selected) >= max_sentences:
            continue
        low = part.lower()
        # prefer sentences that contain one of the query tokens
        if any(token in low for token in lowered_tokens):
            selected.append(part.strip())
    # if none matched, fall back to first sentences
    if not selected:
        for part in parts:
            if part.strip():
                selected.append(part.strip())
            if len(selected) >= max_sentences:
                break
    return selected


def explain_decision(reason_code: str) -> str:
    """Return a concise, factual explanation for a fraud reason code.

    The function is deterministic: the same `reason_code` will always
    yield the same explanation given the same index and chunks.

    No decision logic is applied; the output is an explanation assembled
    from retrieved knowledge chunks and scrubbed of likely PII.
    """
    if not reason_code:
        raise ValueError("reason_code must be a non-empty string")

    model = _load_model()
    index, chunks = _load_index_and_chunks()

    query = _code_to_query(reason_code)
    # deterministic tokenization for extraction
    query_tokens = re.findall(r"\w+", query)

    qvec = model.encode([query], convert_to_numpy=True)
    if qvec.dtype != np.float32:
        qvec = qvec.astype(np.float32)
    faiss.normalize_L2(qvec)

    distances, indices = index.search(qvec, TOP_K)
    # indices shape: (1, TOP_K)
    idx_list = [int(i) for i in indices[0] if i != -1]

    retrieved_texts: List[str] = []
    for idx in idx_list:
        if 0 <= idx < len(chunks):
            retrieved_texts.append(chunks[idx])

    # build explanation deterministically from top results
    summary_sentences: List[str] = []
    for chunk in retrieved_texts:
        sentences = _extract_relevant_sentences(chunk, query_tokens, max_sentences=2)
        for s in sentences:
            cleaned = _scrub_pii(s)
            # avoid duplicates while preserving order
            if cleaned and cleaned not in summary_sentences:
                summary_sentences.append(cleaned)

    if not summary_sentences:
        body = "No relevant guidance found in the knowledge index."
    else:
        body = " ".join(summary_sentences)

    # emphasize behavior patterns rather than individuals
    explanation = (
        f"Explanation for reason code '{reason_code}':\n"
        f"Summary: {body}\n"
        "Behavior patterns referenced: The retrieved guidance focuses on observable transaction and authentication patterns (for example, anomalous transaction amounts, unusual beneficiary additions, or repeated authentication failures) rather than on any individual or account."
    )

    return explanation


if __name__ == "__main__":
    # quick local test if run directly (requires index and chunks present)
    try:
        print(explain_decision("SUS_TRANS_HIGH_VALUE"))
    except Exception as e:
        print("Explainer not runnable: ", e)
