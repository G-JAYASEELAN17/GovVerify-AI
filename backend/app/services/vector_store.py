import json
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss
import numpy as np

from app.config import INDEX_DIR, logger

INDEX_PATH = INDEX_DIR / 'evidence.faiss'
META_PATH = INDEX_DIR / 'metadata.json'
DOCS_CATALOG_PATH = INDEX_DIR / 'docs_catalog.json'

_lock = threading.Lock()
_cached_index: Optional[faiss.Index] = None
_cached_metadata: Optional[List[Dict]] = None


def _load_from_disk() -> Tuple[Optional[faiss.Index], List[Dict]]:
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, []
    try:
        index = faiss.read_index(str(INDEX_PATH))
        metadata = json.loads(META_PATH.read_text(encoding='utf-8'))
        if index.ntotal != len(metadata):
            logger.warning(f"Index count mismatch: {index.ntotal} vectors vs {len(metadata)} metadata items.")
        return index, metadata
    except Exception as e:
        logger.error(f"Error loading FAISS index from disk: {e}")
        return None, []


def load_index() -> Tuple[Optional[faiss.Index], List[Dict]]:
    global _cached_index, _cached_metadata
    with _lock:
        if _cached_index is None or _cached_metadata is None:
            _cached_index, _cached_metadata = _load_from_disk()
        return _cached_index, _cached_metadata


def save_index(index: faiss.Index, metadata: List[Dict]):
    global _cached_index, _cached_metadata
    with _lock:
        faiss.write_index(index, str(INDEX_PATH))
        META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        _cached_index = index
        _cached_metadata = metadata
        logger.info(f"Persistent vector store updated: {len(metadata)} chunks indexed.")


def add_chunks(embeddings: np.ndarray, chunks: List[Dict]) -> int:
    """Appends new chunks to the index."""
    if len(chunks) == 0:
        return 0
    
    index, metadata = load_index()
    emb_array = np.asarray(embeddings, dtype='float32')
    dim = emb_array.shape[1]

    if index is None or index.ntotal == 0:
        index = faiss.IndexFlatIP(dim)
        metadata = []
    
    index.add(emb_array)
    metadata.extend(chunks)
    save_index(index, metadata)
    return len(metadata)


def add_or_replace_document_chunks(doc_id: str, embeddings: np.ndarray, chunks: List[Dict], all_other_embeddings_fn=None) -> int:
    """
    Safely adds chunks for a document. If document already exists, removes old chunks
    and rebuilds the index cleanly to avoid orphaned vectors and desynchronization.
    """
    index, metadata = load_index()
    existing_indices = [i for i, c in enumerate(metadata) if c.get('document_id') == doc_id]

    if not existing_indices:
        # Simple append
        return add_chunks(embeddings, chunks)

    # Document was re-uploaded; rebuild clean index
    remaining_chunks = [c for i, c in enumerate(metadata) if i not in existing_indices]
    remaining_chunks.extend(chunks)
    
    logger.info(f"Rebuilding index for updated document {doc_id}. Total chunks: {len(remaining_chunks)}")
    
    # We will need the embeddings for remaining chunks + new embeddings
    if all_other_embeddings_fn:
        all_embs = all_other_embeddings_fn([c['text'] for c in remaining_chunks])
    else:
        # If no encoder fn passed, use the new embeddings assuming single doc
        all_embs = np.asarray(embeddings, dtype='float32')

    new_index = faiss.IndexFlatIP(all_embs.shape[1])
    new_index.add(np.asarray(all_embs, dtype='float32'))
    save_index(new_index, remaining_chunks)
    return len(remaining_chunks)


def load_doc_catalog() -> List[Dict]:
    if not DOCS_CATALOG_PATH.exists():
        return []
    try:
        return json.loads(DOCS_CATALOG_PATH.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error(f"Error reading document catalog: {e}")
        return []


def save_doc_catalog(catalog: List[Dict]):
    DOCS_CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')


def register_document(doc_record: Dict):
    catalog = load_doc_catalog()
    catalog = [d for d in catalog if d['id'] != doc_record['id'] and d.get('file_hash') != doc_record.get('file_hash')]
    catalog.insert(0, doc_record)
    save_doc_catalog(catalog)
    logger.info(f"Registered document {doc_record['name']} (ID: {doc_record['id']}) in catalog.")


def get_all_documents() -> List[Dict]:
    return load_doc_catalog()


def get_document_by_id(document_id: str) -> Optional[Dict]:
    catalog = load_doc_catalog()
    for d in catalog:
        if d['id'] == document_id or d.get('filename') == document_id or d.get('name') == document_id:
            return d
    return None


def get_document_by_hash(file_hash: str) -> Optional[Dict]:
    catalog = load_doc_catalog()
    for d in catalog:
        if d.get('file_hash') == file_hash:
            return d
    return None
