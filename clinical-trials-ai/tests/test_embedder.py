"""Unit test suite for core/embedder.py."""

import math
from typing import List
import pytest
from sentence_transformers import SentenceTransformer

from core.embedder import embed_query, embed_text, load_model
import config


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class TestEmbedder:
    """Tests for SentenceTransformer model loading and embedding operations."""

    def test_load_model_returns_sentence_transformer(self):
        """Verify load_model returns a genuine SentenceTransformer instance."""
        model = load_model()
        assert isinstance(model, SentenceTransformer)
        assert hasattr(model, "encode")

    def test_load_model_caching(self):
        """Verify consecutive calls to load_model return the exact same cached instance."""
        model_1 = load_model()
        model_2 = load_model()
        assert model_1 is model_2

    def test_load_model_device_cpu(self):
        """Verify the model executes on CPU device for fast, low-overhead inference."""
        model = load_model()
        device_str = str(getattr(model, "device", "")).lower()
        assert "cpu" in device_str

    def test_embed_text_single_text_dimensions_and_types(self):
        """Verify embed_text returns 384-dimensional vector containing Python floats."""
        text = "Phase 3 clinical trial evaluating pembrolizumab in non-small cell lung cancer."
        embeddings = embed_text([text])

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) == 384
        assert all(isinstance(val, float) for val in embeddings[0])

    def test_embed_text_multi_text_batch(self):
        """Verify embed_text generates embeddings for batches of multiple texts."""
        texts = [
            "Chemotherapy regimen for metastatic colorectal cancer.",
            "Novel oral insulin formulation for Type 2 Diabetes Mellitus.",
            "Double-blind placebo-controlled study of cognitive decline in Alzheimer's.",
        ]
        embeddings = embed_text(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 384
            assert all(isinstance(val, float) for val in emb)

    def test_embed_text_empty_list_returns_empty(self):
        """Verify embed_text gracefully returns an empty list when given an empty input."""
        embeddings = embed_text([])
        assert embeddings == []
        assert isinstance(embeddings, list)

    def test_embed_text_with_none_and_empty_strings(self):
        """Verify embed_text safely handles None or empty string elements without crashing."""
        texts = ["Valid clinical trial description", None, "", "   "]
        embeddings = embed_text(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 4
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 384

    def test_embed_query_dimension_and_type(self):
        """Verify embed_query returns a 1D list of 384 floats for a single query."""
        query = "shortness of breath and persistent chest pain"
        embedding = embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(val, float) for val in embedding)

    def test_embed_query_none_and_empty(self):
        """Verify embed_query handles None and empty query strings gracefully."""
        emb_empty = embed_query("")
        assert isinstance(emb_empty, list)
        assert len(emb_empty) == 384

        emb_none = embed_query(None)
        assert isinstance(emb_none, list)
        assert len(emb_none) == 384

    def test_embed_query_and_embed_text_semantic_consistency(self):
        """Verify embed_query(x) produces nearly identical vector to embed_text([x])[0]."""
        text = "Randomized phase 2 trial in hypertensive heart failure"
        vec_query = embed_query(text)
        vec_batch = embed_text([text])[0]

        similarity = _cosine_similarity(vec_query, vec_batch)
        assert similarity > 0.9999, f"Expected near-identical vectors, got cosine {similarity}"

    def test_semantic_discrimination(self):
        """Verify semantic similarity: related medical concepts have higher cosine similarity than unrelated."""
        vec_query = embed_query("joint stiffness and inflammatory arthritis in fingers")
        vec_rheum = embed_query("Clinical evaluation of TNF inhibitors in rheumatoid arthritis patients")
        vec_unrelated = embed_query("Baking artisanal sourdough bread in a traditional brick oven")

        sim_rheum = _cosine_similarity(vec_query, vec_rheum)
        sim_unrelated = _cosine_similarity(vec_query, vec_unrelated)

        assert sim_rheum > sim_unrelated, (
            f"Expected arthritis similarity ({sim_rheum}) > baking similarity ({sim_unrelated})"
        )
