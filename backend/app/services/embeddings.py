from typing import List, Union
import numpy as np

from app.config import EMBEDDING_MODEL, logger
from app.services.providers import get_embedding_provider


EMBEDDING_DIM = 1024


def get_embedding_model():
    """Returns local embedding model if available via provider."""
    provider = get_embedding_provider()
    if hasattr(provider, '_get_model'):
        return provider._get_model()
    return None


def get_model():
    return get_embedding_model()


def release_embedding_model():
    """Explicitly releases BGE-M3 from RAM/memory."""
    provider = get_embedding_provider()
    provider.release()


def encode(texts: Union[str, List[str]], batch_size: int = 4, release_after: bool = False) -> np.ndarray:
    """
    Generates normalized dense vector embeddings with small CPU-friendly batch sizes.
    Routes to Local or Remote BGE-M3 provider according to deployment environment.
    Always returns float32 unit-normalized numpy array.
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.empty((0, get_embedding_dimension()), dtype='float32')

    provider = get_embedding_provider()
    embeddings = provider.encode(texts, batch_size=batch_size)

    if release_after:
        release_embedding_model()

    return np.asarray(embeddings, dtype='float32')


def get_embedding_dimension() -> int:
    return EMBEDDING_DIM

