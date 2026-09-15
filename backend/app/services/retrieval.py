from typing import List, Dict, Optional
import numpy as np

from app.config import TOP_K, SIMILARITY_THRESHOLD, logger
from app.services.embeddings import encode
from app.services.vector_store import load_index


def search(
    query: str,
    k: int = TOP_K,
    min_score: Optional[float] = None,
    document_id: Optional[str] = None
) -> List[Dict]:
    """
    Performs semantic vector search over the persistent FAISS index.
    Filters candidates below the minimum relevance threshold.
    """
    threshold = min_score if min_score is not None else SIMILARITY_THRESHOLD
    index, metadata = load_index()
    
    if index is None or not metadata or index.ntotal == 0:
        logger.warning("Search called on empty vector index.")
        return []

    # Encode query with normalized embedding
    q_emb = encode(query).astype('float32')
    
    # Query up to min(k * 2, total) to allow filtering if document_id or threshold is applied
    fetch_k = min(max(k * 2, 10), len(metadata))
    scores, indices = index.search(q_emb, fetch_k)

    results: List[Dict] = []
    
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        
        sim_score = float(score)
        chunk = dict(metadata[idx])
        chunk['score'] = sim_score

        # Optional document isolation filter
        if document_id and chunk.get('document_id') != document_id:
            continue

        # Check relevance threshold
        if sim_score >= threshold:
            results.append(chunk)

        if len(results) >= k:
            break

    if results:
        top = results[0]
        logger.info(f"[RETRIEVAL] query='{query[:60]}...' top_doc='{top.get('document_name')}' page={top.get('page_number')} score={top['score']:.4f}")
    else:
        logger.info(f"[RETRIEVAL] query='{query[:60]}...' NO_RELEVANT_EVIDENCE (all candidates below threshold {threshold:.2f})")

    return results


def search_with_diagnostics(query: str, k: int = 5) -> Dict:
    """Returns all top candidate results with raw scores, threshold, and acceptance flag for debugging."""
    index, metadata = load_index()
    if index is None or not metadata:
        return {'query': query, 'threshold': SIMILARITY_THRESHOLD, 'total_candidates': 0, 'relevant_hits': []}

    q_emb = encode(query).astype('float32')
    fetch_k = min(k, len(metadata))
    scores, indices = index.search(q_emb, fetch_k)

    hits = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        sim_score = round(float(score), 4)
        hits.append({
            'rank': rank,
            'document': chunk.get('document_name', 'Unknown'),
            'page': chunk.get('page_number', 0),
            'score': sim_score,
            'threshold': SIMILARITY_THRESHOLD,
            'accepted': sim_score >= SIMILARITY_THRESHOLD,
            'text': chunk.get('text', '')[:200]
        })

    return {
        'query': query,
        'threshold': SIMILARITY_THRESHOLD,
        'total_candidates': len(metadata),
        'relevant_hits': hits
    }

