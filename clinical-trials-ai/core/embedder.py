"""Embedding service using SentenceTransformers (all-MiniLM-L6-v2)."""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

# Module-level cache for the loaded model
_MODEL: Optional[SentenceTransformer] = None


def load_model() -> SentenceTransformer:
    """Load or retrieve the cached SentenceTransformer instance.

    Uses EMBEDDING_MODEL and EMBEDDING_DEVICE from config.py.
    The instance is cached at module level to prevent re-initialization overhead.
    """
    global _MODEL
    if _MODEL is None:
        model_name = getattr(config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        device = getattr(config, "EMBEDDING_DEVICE", "cpu")
        logger.info("Loading SentenceTransformer model '%s' on device '%s'", model_name, device)
        _MODEL = SentenceTransformer(model_name, device=device)
    return _MODEL


def embed_text(texts: List[str]) -> List[List[float]]:
    """Generate dense embedding vectors for a list of strings.

    Args:
        texts: List of input text strings to embed.

    Returns:
        List of 384-dimensional float vectors (shape N x 384).
        Returns an empty list if input texts is empty.
    """
    if not texts:
        return []

    model = load_model()
    # Normalize input items to strings defensively
    clean_texts = [str(t) if t is not None else "" for t in texts]

    embeddings = model.encode(clean_texts, convert_to_numpy=True)
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(float(x) for x in emb) for emb in embeddings]


def embed_query(query: str) -> List[float]:
    """Generate a 384-dimensional dense vector for a single query string.

    Args:
        query: Query string (e.g. user-entered symptoms).

    Returns:
        List of 384 floats representing the embedding vector.
    """
    clean_query = str(query) if query is not None else ""
    model = load_model()
    embedding = model.encode(clean_query, convert_to_numpy=True)
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return [float(x) for x in embedding]
