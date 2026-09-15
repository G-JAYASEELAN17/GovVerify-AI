import gc
from threading import Lock
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, logger, get_process_memory_mb


_model: SentenceTransformer | None = None
_model_lock = Lock()
EMBEDDING_DIM = 1024


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            logger.info(f"[GovVerify] RAM before BGE load: {get_process_memory_mb()} MB")
            logger.info(f"[GovVerify] Loading BGE-M3 model on CPU: {EMBEDDING_MODEL}")
            model = SentenceTransformer(
                EMBEDDING_MODEL,
                device='cpu',
                model_kwargs={'low_cpu_mem_usage': True}
            )
            model.eval()
            try:
                # Dynamic INT8 quantization for CPU linear layers (reduces RAM footprint)
                if hasattr(model, '_modules') and '0' in model._modules:
                    sub = model._modules['0']
                    if hasattr(sub, 'auto_model'):
                        sub.auto_model = torch.quantization.quantize_dynamic(
                            sub.auto_model, {torch.nn.Linear}, dtype=torch.qint8
                        )
            except Exception as q_err:
                logger.debug(f"BGE-M3 dynamic quantization note: {q_err}")

            _model = model
            gc.collect()
            logger.info(f"[GovVerify] RAM after BGE load: {get_process_memory_mb()} MB")
    return _model


def get_model() -> SentenceTransformer:
    return get_embedding_model()


def release_embedding_model():
    """Explicitly frees BGE-M3 from RAM to avoid concurrent memory overlap with DeBERTa."""
    global _model
    with _model_lock:
        if _model is not None:
            logger.info("Releasing BGE-M3 embedding model from RAM...")
            del _model
            _model = None
            gc.collect()
            logger.info(f"[GovVerify] RAM after cleanup: {get_process_memory_mb()} MB")


def encode(texts: Union[str, List[str]], batch_size: int = 4, release_after: bool = False) -> np.ndarray:
    """
    Generates normalized dense vector embeddings with small CPU-friendly batch sizes.
    Always returns float32 unit-normalized numpy array.
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.empty((0, get_embedding_dimension()), dtype='float32')

    model = get_embedding_model()
    logger.info(f"[GovVerify] RAM before embedding: {get_process_memory_mb()} MB")
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    logger.info(f"[GovVerify] RAM after embedding: {get_process_memory_mb()} MB")
    
    if release_after:
        release_embedding_model()

    return np.asarray(embeddings, dtype='float32')


def get_embedding_dimension() -> int:
    global _model
    if _model is not None:
        return _model.get_sentence_embedding_dimension()
    return EMBEDDING_DIM

