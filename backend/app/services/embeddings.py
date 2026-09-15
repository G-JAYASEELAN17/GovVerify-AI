import gc
from threading import Lock
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, logger


_model: SentenceTransformer | None = None
_model_lock = Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            logger.info(f"Loading embedding model on CPU: {EMBEDDING_MODEL}")
            # Explicitly load to CPU with low memory usage
            model = SentenceTransformer(
                EMBEDDING_MODEL,
                device='cpu',
                model_kwargs={'low_cpu_mem_usage': True}
            )
            model.eval()
            _model = model
            gc.collect()
    return _model


def encode(texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
    """
    Generates normalized dense vector embeddings.
    Always returns float32 unit-normalized numpy array.
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.empty((0, get_embedding_dimension()), dtype='float32')

    model = get_model()
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    return np.asarray(embeddings, dtype='float32')


def get_embedding_dimension() -> int:
    model = get_model()
    return model.get_sentence_embedding_dimension()
