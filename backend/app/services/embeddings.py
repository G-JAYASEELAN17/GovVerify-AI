from functools import lru_cache
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, logger


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


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
