# rag.py — Lightweight policy retrieval via TF-IDF + cosine similarity
#
# Design note — why not ChromaDB + neural embeddings?
#   The notebook uses OpenAIEmbeddings. Our first production swap tried
#   HuggingFace sentence-transformers (PyTorch, ~530 MB) then FastEmbed
#   (ONNX, ~200 MB). Both exceed Render free tier's 512 MB RAM limit.
#
#   TF-IDF + cosine similarity (sklearn) achieves equivalent retrieval quality
#   for this small, domain-specific corpus, with <5 MB memory overhead.
#   The public interface (create_policy_store / retrieve_relevant_policies)
#   is unchanged so agents.py needs no modification.

import re
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Resolve path to underwriting_policies.pdf relative to this file
_POLICY_PDF = Path(__file__).parent.parent / "data" / "underwriting_policies.pdf"


@dataclass
class _Doc:
    """Minimal document wrapper matching the LangChain Document interface
    so retrieve_relevant_policies() works identically to the ChromaDB version."""
    page_content: str


class TFIDFPolicyStore:
    """In-memory policy store using TF-IDF vectors and cosine similarity.

    Drop-in replacement for a ChromaDB vectorstore: exposes the same
    similarity_search(query, k) method used by retrieve_relevant_policies().
    """

    def __init__(self, chunks: list[str]):
        self._chunks = chunks
        # Unigrams + bigrams capture phrases like "debt-to-income ratio"
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(chunks)

    def similarity_search(self, query: str, k: int = 6) -> list[_Doc]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        top_indices = np.argsort(scores)[::-1][:k]
        return [_Doc(page_content=self._chunks[i]) for i in top_indices]


def create_policy_store(pdf_path: str | None = None) -> TFIDFPolicyStore:
    """Load underwriting_policies.pdf into an in-memory TF-IDF policy store.

    Args:
        pdf_path: Optional override for PDF location. Defaults to data/underwriting_policies.pdf.

    Returns:
        TFIDFPolicyStore ready for similarity_search queries.
    """
    path = pdf_path or str(_POLICY_PDF)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Policy PDF not found at {path}. "
            "Ensure underwriting_policies.pdf is in the data/ directory."
        )

    # Load PDF pages
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Chunk into 1000-char pieces with 200-char overlap (matches notebook settings)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    policy_chunks = text_splitter.split_documents(documents)

    chunk_texts = [doc.page_content for doc in policy_chunks]
    return TFIDFPolicyStore(chunk_texts)


def retrieve_relevant_policies(query: str, vectorstore: TFIDFPolicyStore) -> str:
    """Retrieve and deduplicate policy chunks relevant to the query.

    Args:
        query: Natural language question about underwriting policy.
        vectorstore: TFIDFPolicyStore returned by create_policy_store().

    Returns:
        Deduplicated policy text, grouped by section heading.
    """
    docs = vectorstore.similarity_search(query, k=6)

    section_map: dict[str, str] = {}

    for doc in docs:
        text = doc.page_content.strip()

        # Group by section heading (e.g. "2.3 Self-Employment Income")
        match = re.match(r"^\d+\.\d+\s+[A-Za-z ].+", text)
        section = match.group(0) if match else "OTHER"

        if section not in section_map:
            section_map[section] = text
        elif text not in section_map[section]:
            section_map[section] += "\n" + text

    return "\n\n".join(section_map.values())
