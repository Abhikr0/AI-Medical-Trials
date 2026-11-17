"""Unit test suite for core/vector_store.py."""

from pathlib import Path
from typing import List
import chromadb
import pytest

from core.vector_store import (
    build_where_clause,
    clear_collection,
    get_chroma_client,
    get_collection,
    get_trial_count,
    sanitize_metadata,
    search_trials,
    upsert_trials,
)
from tests.conftest import make_deterministic_vector


class TestMetadataSanitization:
    """Tests for metadata sanitization helper."""

    def test_sanitize_none_to_empty_string(self):
        """Verify None values are converted to empty strings."""
        meta = {"phase": None, "sponsor": None, "title": "Trial Title"}
        sanitized = sanitize_metadata(meta)
        assert sanitized["phase"] == ""
        assert sanitized["sponsor"] == ""
        assert sanitized["title"] == "Trial Title"

    def test_sanitize_list_joining(self):
        """Verify list and sequence types are converted to comma-separated strings."""
        meta = {
            "conditions": ["Cancer", "Leukemia"],
            "locations": ["Boston", "New York", "Chicago"],
            "empty_list": [],
        }
        sanitized = sanitize_metadata(meta)
        assert sanitized["conditions"] == "Cancer, Leukemia"
        assert sanitized["locations"] == "Boston, New York, Chicago"
        assert sanitized["empty_list"] == ""

    def test_sanitize_primitive_preservation(self):
        """Verify standard primitives (int, float, bool, str) are preserved."""
        meta = {
            "nct_id": "NCT01234567",
            "enrollment": 150,
            "score": 98.6,
            "is_interventional": True,
            "is_active": False,
        }
        sanitized = sanitize_metadata(meta)
        assert sanitized["nct_id"] == "NCT01234567"
        assert sanitized["enrollment"] == 150
        assert sanitized["score"] == 98.6
        assert sanitized["is_interventional"] is True
        assert sanitized["is_active"] is False

    def test_sanitize_nested_dict_and_non_dict_inputs(self):
        """Verify nested structures are safely converted and non-dict inputs return empty dict."""
        meta = {"details": {"sub_key": 123}}
        sanitized = sanitize_metadata(meta)
        assert "{'sub_key': 123}" in sanitized["details"]

        assert sanitize_metadata(None) == {}
        assert sanitize_metadata("not a dict") == {}


class TestWhereClauseBuilder:
    """Tests for ChromaDB where filter builder."""

    def test_build_where_clause_none_or_all(self):
        """Verify None or 'All' returns None (never empty dict which causes ChromaDB error)."""
        assert build_where_clause(None, None) is None
        assert build_where_clause("All", "All") is None
        assert build_where_clause("", "") is None
        assert build_where_clause("   ", None) is None

    def test_build_where_clause_phase_only(self):
        """Verify single phase filter returns single equality operator."""
        clause = build_where_clause(phase="Phase 2")
        assert clause == {"phase": {"$eq": "Phase 2"}}

        clause_with_all = build_where_clause(phase="Phase 3", status="All")
        assert clause_with_all == {"phase": {"$eq": "Phase 3"}}

    def test_build_where_clause_status_only(self):
        """Verify single status filter returns single equality operator."""
        clause = build_where_clause(status="RECRUITING")
        assert clause == {"status": {"$eq": "RECRUITING"}}

        clause_with_all = build_where_clause(phase="All", status="COMPLETED")
        assert clause_with_all == {"status": {"$eq": "COMPLETED"}}

    def test_build_where_clause_both_filters(self):
        """Verify both filters combine via $and operator."""
        clause = build_where_clause(phase="Phase 3", status="RECRUITING")
        assert clause == {
            "$and": [
                {"phase": {"$eq": "Phase 3"}},
                {"status": {"$eq": "RECRUITING"}},
            ]
        }


class TestVectorStoreOperations:
    """Tests for ChromaDB client, collection, upsert, and search operations."""

    def test_get_chroma_client_and_collection(self, tmp_path):
        """Verify persistent client initializes directory and collection with cosine space."""
        db_path = str(tmp_path / "chroma_test_store")
        client = get_chroma_client(path=db_path)
        assert client is not None
        assert isinstance(client, chromadb.api.ClientAPI)
        assert hasattr(client, "get_or_create_collection")
        assert Path(db_path).exists()

        collection = get_collection(client=client, collection_name="test_collection")
        assert collection.name == "test_collection"
        meta = collection.metadata or {}
        assert meta.get("hnsw:space") == "cosine"

    def test_upsert_trials_populates_collection(self, isolated_persistent_chroma):
        """Verify upserting trials increases collection count and stores metadata."""
        col = isolated_persistent_chroma["collection"]
        initial_count = get_trial_count(collection=col)

        ids = ["NCT_TEST_01", "NCT_TEST_02"]
        embeddings = [make_deterministic_vector(doc) for doc in ["Doc 1", "Doc 2"]]
        documents = ["Cardiology study on heart failure", "Oncology trial on melanoma"]
        metadatas = [
            {"phase": "Phase 1", "status": "RECRUITING", "conditions": ["Heart Failure"]},
            {"phase": "Phase 2", "status": "COMPLETED", "conditions": ["Melanoma"]},
        ]

        upsert_trials(ids, embeddings, documents, metadatas, collection=col)
        assert get_trial_count(collection=col) == initial_count + 2

    def test_upsert_empty_list_is_noop(self, isolated_persistent_chroma):
        """Verify passing empty lists to upsert_trials does not raise and does not alter count."""
        col = isolated_persistent_chroma["collection"]
        initial_count = get_trial_count(collection=col)
        upsert_trials([], [], [], [], collection=col)
        assert get_trial_count(collection=col) == initial_count

    def test_upsert_idempotency_and_update(self, isolated_persistent_chroma):
        """Verify re-upserting same IDs updates records without increasing total count."""
        col = isolated_persistent_chroma["collection"]
        ids = ["NCT_IDEMPOTENT"]
        embeddings = [make_deterministic_vector("Initial")]
        documents = ["Initial document"]
        metadatas = [{"phase": "Phase 1", "status": "RECRUITING"}]

        upsert_trials(ids, embeddings, documents, metadatas, collection=col)
        count_first = get_trial_count(collection=col)

        # Update same ID
        upsert_trials(
            ids,
            [make_deterministic_vector("Updated")],
            ["Updated document"],
            [{"phase": "Phase 2", "status": "COMPLETED"}],
            collection=col,
        )
        assert get_trial_count(collection=col) == count_first

        # Verify updated values
        res = col.get(ids=ids)
        assert res["documents"][0] == "Updated document"
        assert res["metadatas"][0]["status"] == "COMPLETED"
        assert res["metadatas"][0]["phase"] == "Phase 2"

    def test_upsert_duplicate_ids_within_batch_deduplication(self, isolated_persistent_chroma):
        """Verify duplicate IDs in a single batch are cleanly deduplicated, preserving latest record."""
        col = isolated_persistent_chroma["collection"]
        ids = ["NCT_BATCH_DUP", "NCT_BATCH_DUP"]
        docs = ["Version 1", "Version 2 (latest)"]
        embeddings = [make_deterministic_vector(d) for d in docs]
        metadatas = [
            {"phase": "Phase 1", "status": "RECRUITING"},
            {"phase": "Phase 3", "status": "COMPLETED"},
        ]

        upsert_trials(ids, embeddings, docs, metadatas, collection=col)
        assert get_trial_count(collection=col) == 1

        res = col.get(ids=["NCT_BATCH_DUP"])
        assert res["documents"][0] == "Version 2 (latest)"
        assert res["metadatas"][0]["phase"] == "Phase 3"
        assert res["metadatas"][0]["status"] == "COMPLETED"

    def test_upsert_mismatched_batch_lengths_raises_error(self, isolated_persistent_chroma):
        """Verify passing mismatched batch lengths raises ValueError."""
        col = isolated_persistent_chroma["collection"]
        with pytest.raises(ValueError):
            upsert_trials(
                ids=["NCT_M1", "NCT_M2"],
                embeddings=[make_deterministic_vector("x")],
                documents=["Doc 1", "Doc 2"],
                metadatas=[{"phase": "Phase 1"}, {"phase": "Phase 2"}],
                collection=col,
            )

    def test_search_trials_cosine_ranking(self, isolated_persistent_chroma):
        """Verify search_trials returns items ordered by ascending cosine distance."""
        col = isolated_persistent_chroma["collection"]
        ids = ["NCT_SIM_A", "NCT_SIM_B", "NCT_SIM_C"]
        docs = ["Target document", "Somewhat similar document", "Completely unrelated text"]
        embeddings = [make_deterministic_vector(d) for d in docs]
        metadatas = [{"phase": "Phase 1", "status": "RECRUITING"} for _ in docs]

        upsert_trials(ids, embeddings, docs, metadatas, collection=col)

        query_vec = make_deterministic_vector("Target document")
        res = search_trials(query_vec, n_results=3, collection=col)

        assert res["ids"][0][0] == "NCT_SIM_A"
        distances = res["distances"][0]
        assert distances[0] <= distances[1] <= distances[2]

    def test_search_trials_filtering(self, isolated_persistent_chroma):
        """Verify phase and status filters in search_trials."""
        col = isolated_persistent_chroma["collection"]
        upsert_trials(
            ids=["NCT_P1_R", "NCT_P1_C", "NCT_P3_R", "NCT_P3_C"],
            embeddings=[make_deterministic_vector(f"item {i}") for i in range(4)],
            documents=[f"Trial item {i}" for i in range(4)],
            metadatas=[
                {"phase": "Phase 1", "status": "RECRUITING"},
                {"phase": "Phase 1", "status": "COMPLETED"},
                {"phase": "Phase 3", "status": "RECRUITING"},
                {"phase": "Phase 3", "status": "COMPLETED"},
            ],
            collection=col,
        )

        query_vec = make_deterministic_vector("item 0")

        # Phase filter only
        res_p1 = search_trials(query_vec, n_results=10, phase="Phase 1", collection=col)
        assert len(res_p1["metadatas"][0]) == 2
        assert all(m["phase"] == "Phase 1" for m in res_p1["metadatas"][0])

        # Status filter only
        res_rec = search_trials(query_vec, n_results=10, status="RECRUITING", collection=col)
        assert len(res_rec["metadatas"][0]) == 2
        assert all(m["status"] == "RECRUITING" for m in res_rec["metadatas"][0])

        # Both phase and status filters
        res_both = search_trials(
            query_vec, n_results=10, phase="Phase 3", status="COMPLETED", collection=col
        )
        assert len(res_both["metadatas"][0]) == 1
        assert res_both["metadatas"][0][0]["phase"] == "Phase 3"
        assert res_both["metadatas"][0][0]["status"] == "COMPLETED"

        # "All" filter should return all
        res_all = search_trials(
            query_vec, n_results=10, phase="All", status="All", collection=col
        )
        assert len(res_all["metadatas"][0]) == 4

    def test_search_trials_empty_collection(self, tmp_path):
        """Verify search on an empty collection returns structured empty results without crashing."""
        db_path = str(tmp_path / "empty_chroma")
        client = chromadb.PersistentClient(path=db_path)
        col = client.create_collection("empty_col", metadata={"hnsw:space": "cosine"})

        query_vec = make_deterministic_vector("query")
        res = search_trials(query_vec, n_results=5, collection=col)

        assert "ids" in res
        assert res["ids"] == [[]]
        assert res["documents"] == [[]]
        assert res["metadatas"] == [[]]
        assert res["distances"] == [[]]

    def test_clear_collection(self, isolated_persistent_chroma):
        """Verify clear_collection removes all records from the collection."""
        col = isolated_persistent_chroma["collection"]
        upsert_trials(
            ids=["NCT_CLEAR_1", "NCT_CLEAR_2"],
            embeddings=[make_deterministic_vector("c1"), make_deterministic_vector("c2")],
            documents=["Clear doc 1", "Clear doc 2"],
            metadatas=[{"phase": "Phase 1"}, {"phase": "Phase 2"}],
            collection=col,
        )
        assert get_trial_count(collection=col) == 2

        clear_collection(collection=col)
        assert get_trial_count(collection=col) == 0

    def test_persistence_across_client_reinitialization(self, tmp_path):
        """Verify data persists when a new PersistentClient is opened on the same storage path."""
        db_path = str(tmp_path / "persistent_storage_dir")

        # Step 1: Open client 1, insert trial
        client_1 = get_chroma_client(path=db_path)
        col_1 = get_collection(client=client_1, collection_name="persisted_trials")
        target_vec = make_deterministic_vector("Persisted clinical study")
        upsert_trials(
            ids=["NCT_PERSIST_1"],
            embeddings=[target_vec],
            documents=["Long-term survival outcomes in immuno-oncology."],
            metadatas=[{"phase": "Phase 3", "status": "ACTIVE_NOT_RECRUITING"}],
            collection=col_1,
        )
        assert col_1.count() == 1

        # Step 2: Open client 2 on the exact same directory (simulating app restart)
        client_2 = chromadb.PersistentClient(path=db_path)
        col_2 = client_2.get_collection(name="persisted_trials")
        assert col_2.count() == 1

        res = search_trials(target_vec, n_results=1, collection=col_2)
        assert res["ids"][0][0] == "NCT_PERSIST_1"
        assert res["documents"][0][0] == "Long-term survival outcomes in immuno-oncology."
        assert res["metadatas"][0][0]["status"] == "ACTIVE_NOT_RECRUITING"
