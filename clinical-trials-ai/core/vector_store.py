"""Vector storage and similarity search module using ChromaDB."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb

import config


def get_chroma_client(path: Optional[str] = None) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client for CHROMA_DB_PATH or specified path.

    Args:
        path: Optional override path for ChromaDB storage directory.

    Returns:
        chromadb.PersistentClient instance connected to storage directory.
    """
    target_path = path or getattr(config, "CHROMA_DB_PATH", "./data/chroma_db")
    Path(target_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(target_path))


def get_collection(
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: Optional[str] = None,
) -> chromadb.Collection:
    """Return the clinical_trials collection configured with cosine distance metric.

    Args:
        client: Optional ChromaDB client instance (default: get_chroma_client()).
        collection_name: Optional collection name override (default: config.COLLECTION_NAME).

    Returns:
        chromadb.Collection configured for cosine distance space.
    """
    if client is None:
        client = get_chroma_client()
    target_name = collection_name or getattr(config, "COLLECTION_NAME", "clinical_trials")
    return client.get_or_create_collection(
        name=target_name,
        metadata={"hnsw:space": "cosine"},
    )


def sanitize_metadata(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize metadata dictionary for ChromaDB compatibility.

    ChromaDB rejects None values and nested collections.
    - None values are mapped to empty strings ("").
    - Lists/tuples/sets are joined to comma-separated strings.
    - Primitives (int, float, str, bool) are preserved.
    - Other types are converted to str.

    Args:
        meta: Input metadata dictionary.

    Returns:
        Sanitized metadata dictionary with primitive or string values.
    """
    if not isinstance(meta, dict):
        return {}

    sanitized: Dict[str, Any] = {}
    for k, v in meta.items():
        key_str = str(k)
        if v is None:
            sanitized[key_str] = ""
        elif isinstance(v, bool):
            sanitized[key_str] = v
        elif isinstance(v, (int, float, str)):
            sanitized[key_str] = v
        elif isinstance(v, (list, tuple, set)):
            sanitized[key_str] = ", ".join(str(item) for item in v if item is not None)
        else:
            sanitized[key_str] = str(v)
    return sanitized


def build_where_clause(
    phase: Optional[str] = None,
    status: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build a ChromaDB-compliant `where` filter dictionary.

    ChromaDB throws a ValueError when passed an empty filter (`where={}`).
    This helper returns None when no filter is active.

    Args:
        phase: Trial phase filter (e.g. 'Phase 2' or 'All' / None).
        status: Recruitment status filter (e.g. 'RECRUITING' or 'All' / None).

    Returns:
        A valid ChromaDB where clause dictionary, or None if no active filters.
    """
    conditions: List[Dict[str, Any]] = []

    if phase and phase != "All" and str(phase).strip() and str(phase).strip() != "All":
        conditions.append({"phase": {"$eq": str(phase).strip()}})

    if status and status != "All" and str(status).strip() and str(status).strip() != "All":
        conditions.append({"status": {"$eq": str(status).strip()}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def upsert_trials(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    collection: Optional[chromadb.Collection] = None,
) -> None:
    """Upsert sanitized trials into ChromaDB collection.

    Gracefully handles empty inputs as a no-op to avoid ChromaDB ValueError.
    Deduplicates IDs within the batch by preserving the latest record per ID.

    Args:
        ids: Unique identifiers (e.g. NCT IDs).
        embeddings: 384-dimensional dense vectors.
        documents: Searchable text representations of the trials.
        metadatas: Trial metadata dictionaries.
        collection: Optional ChromaDB collection (default: get_collection()).
    """
    if not ids:
        return

    if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
        raise ValueError("Mismatched batch lengths in upsert_trials")

    if collection is None:
        collection = get_collection()

    sanitized_metadatas = []
    for i, meta in enumerate(metadatas):
        s_meta = sanitize_metadata(meta)
        # ChromaDB requires metadata dicts to be non-empty if metadatas is provided
        if not s_meta:
            s_meta = {"nct_id": ids[i] if i < len(ids) else ""}
        sanitized_metadatas.append(s_meta)

    # Convert embeddings to lists of floats if numpy arrays or tensors are provided
    clean_embeddings = []
    for emb in embeddings:
        if hasattr(emb, "tolist"):
            clean_embeddings.append(emb.tolist())
        else:
            clean_embeddings.append([float(x) for x in emb])

    # Deduplicate within batch preserving the latest occurrence per ID
    seen_ids = set()
    deduped_indices = []
    for idx in reversed(range(len(ids))):
        id_val = ids[idx]
        if id_val not in seen_ids:
            seen_ids.add(id_val)
            deduped_indices.append(idx)
    deduped_indices.reverse()

    deduped_ids = [ids[i] for i in deduped_indices]
    deduped_embeddings = [clean_embeddings[i] for i in deduped_indices]
    deduped_documents = [str(documents[i]) for i in deduped_indices]
    deduped_metadatas = [sanitized_metadatas[i] for i in deduped_indices]

    collection.upsert(
        ids=deduped_ids,
        embeddings=deduped_embeddings,
        documents=deduped_documents,
        metadatas=deduped_metadatas,
    )


def search_trials(
    query_embedding: List[float],
    n_results: int = 10,
    phase: Optional[str] = None,
    status: Optional[str] = None,
    collection: Optional[chromadb.Collection] = None,
) -> Dict[str, Any]:
    """Query top-K trials with optional phase and status filtering.

    Args:
        query_embedding: 384-dimensional query vector.
        n_results: Number of nearest neighbors to retrieve.
        phase: Optional phase filter string (or 'All').
        status: Optional recruitment status filter string (or 'All').
        collection: Optional ChromaDB collection (default: get_collection()).

    Returns:
        Dictionary containing 'ids', 'documents', 'metadatas', and 'distances'.
    """
    if collection is None:
        collection = get_collection()

    safe_n_results = max(1, int(n_results))

    # Convert query embedding to list of floats if needed
    clean_query = (
        query_embedding.tolist()
        if hasattr(query_embedding, "tolist")
        else [float(x) for x in query_embedding]
    )

    where_filter = build_where_clause(phase=phase, status=status)
    kwargs: Dict[str, Any] = {
        "query_embeddings": [clean_query],
        "n_results": safe_n_results,
    }
    if where_filter is not None:
        kwargs["where"] = where_filter

    return collection.query(**kwargs)


def get_trial_count(collection: Optional[chromadb.Collection] = None) -> int:
    """Return total number of trials indexed in ChromaDB.

    Args:
        collection: Optional ChromaDB collection (default: get_collection()).

    Returns:
        Integer count of indexed records.
    """
    if collection is None:
        collection = get_collection()
    return collection.count()


def clear_collection(collection: Optional[chromadb.Collection] = None) -> None:
    """Clear/delete all records in the collection.

    Args:
        collection: Optional ChromaDB collection (default: clear active collection).
    """
    if collection is None:
        client = get_chroma_client()
        target_name = getattr(config, "COLLECTION_NAME", "clinical_trials")
        try:
            client.delete_collection(name=target_name)
        except Exception:
            pass
        # Recreate the collection with cosine distance
        client.get_or_create_collection(
            name=target_name,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        try:
            existing = collection.get()
            ids = existing.get("ids", [])
            if ids:
                collection.delete(ids=ids)
        except Exception:
            pass
