import numpy as np
from vertexai.language_models import TextEmbeddingModel
import vertexai
import os
import json

from knowledge_base_data import KNOWLEDGE_BASE
from config import GCP_PROJECT_ID, GCP_LOCATION


_embedding_model = None
_embedded_knowledge_base = []


def initialize_embedding_model():
    """Initializes the Vertex AI TextEmbeddingModel."""
    global _embedding_model
    try:

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

        _embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        print(
            "Vertex AI TextEmbeddingModel 'text-embedding-004' initialized successfully."
        )
    except Exception as e:
        print(f"ERROR: Could not initialize Vertex AI TextEmbeddingModel: {e}")
        print(
            "Please ensure 'gcloud auth application-default login' is run and your GCP_PROJECT_ID/GCP_LOCATION are correct in config.py."
        )
        _embedding_model = None


def initialize_embeddings():
    """
    Generates embeddings for all documents in the KNOWLEDGE_BASE and stores them in-memory.
    This function should be called once at application startup.
    """
    if _embedding_model is None:
        print(
            "WARNING: Embedding model is not initialized. Cannot initialize embeddings for RAG. RAG will not function."
        )
        return

    print(
        "Initializing RAG knowledge base embeddings (this may take a moment, depending on KNOWLEDGE_BASE size)..."
    )
    global _embedded_knowledge_base
    _embedded_knowledge_base = []

    documents_to_embed = [doc["content"] for doc in KNOWLEDGE_BASE]

    try:
        for i, doc_content in enumerate(documents_to_embed):
            embeddings = _embedding_model.get_embeddings([doc_content])[0].values
            _embedded_knowledge_base.append(
                {"original_content": doc_content, "embedding": np.array(embeddings)}
            )
            if (i + 1) % 10 == 0:
                print(f"  ... Embedded {i + 1}/{len(documents_to_embed)} documents.")

        print(f"Finished embedding {len(_embedded_knowledge_base)} documents for RAG.")
    except Exception as e:
        print(f"ERROR: Failed to embed knowledge base documents: {e}")
        print("RAG will not function. Check API quotas, network, or content issues.")
        _embedded_knowledge_base = []


def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0

    return dot_product / (norm_vec1 * norm_vec2)


def retrieve_documents(query: str, top_n: int = 3) -> list:
    """
    Retrieves top_n most semantically similar documents from the embedded knowledge base
    based on the user query.
    """
    if _embedding_model is None or not _embedded_knowledge_base:
        print(
            "WARNING: Embedding model or embedded knowledge base not ready. RAG will return empty context."
        )
        return []

    try:
        query_embedding_values = _embedding_model.get_embeddings([query])[0].values
        query_embedding = np.array(query_embedding_values)

        similarities = []
        for i, doc_data in enumerate(_embedded_knowledge_base):
            similarity = cosine_similarity(query_embedding, doc_data["embedding"])
            similarities.append((similarity, doc_data["original_content"]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        print(f"Top {top_n} document similarities for query '{query}':")
        for sim, content in similarities[:top_n]:
            print(f"  Similarity: {sim:.4f} - Content: '{content[:50]}...'")

        top_docs_content = [doc_content for sim, doc_content in similarities[:top_n]]
        return top_docs_content

    except Exception as e:
        print(f"ERROR: Failed during embedding retrieval for query '{query}': {e}")
        return []
