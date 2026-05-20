# rag.py — Policy retrieval via ChromaDB + HuggingFace embeddings
#
# Embedding swap from notebook:
#   Notebook used OpenAIEmbeddings (requires paid API key).
#   Here we use HuggingFaceEmbeddings (all-MiniLM-L6-v2) — free, no API key,
#   runs locally via sentence-transformers.

import re
import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Resolve path to underwriting_policies.pdf relative to this file
_POLICY_PDF = Path(__file__).parent.parent / "data" / "underwriting_policies.pdf"


def create_policy_store(pdf_path: str | None = None):
    """Load underwriting_policies.pdf into an in-memory ChromaDB vector store.

    Args:
        pdf_path: Optional override for PDF location. Defaults to data/underwriting_policies.pdf.

    Returns:
        ChromaDB vector store ready for similarity_search queries.
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

    # HuggingFace embeddings — free, local, no API key required
    # all-MiniLM-L6-v2 is small (80 MB), fast, and accurate enough for this use case
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Build in-memory ChromaDB collection
    vectorstore = Chroma.from_documents(
        documents=policy_chunks,
        embedding=embeddings,
        collection_name="underwriting_policies",
    )

    return vectorstore


def retrieve_relevant_policies(query: str, vectorstore) -> str:
    """Retrieve and deduplicate policy chunks relevant to the query.

    Args:
        query: Natural language question about underwriting policy.
        vectorstore: ChromaDB instance returned by create_policy_store().

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
