from abc import ABC, abstractmethod
import gc
from threading import Lock
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import httpx
import torch

from app.config import (
    EMBEDDING_MODEL,
    NLI_MODEL,
    INFERENCE_PROVIDER,
    HF_API_KEY,
    EMBEDDING_API_URL,
    NLI_API_URL,
    INFERENCE_TIMEOUT_SEC,
    logger,
    get_process_memory_mb,
    get_container_memory_limit_mb,
    check_safe_memory_for_model,
    InsufficientMemoryError
)


# ==============================================================================
# Embedding Providers (BAAI/bge-m3)
# ==============================================================================

class EmbeddingProvider(ABC):
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], batch_size: int = 4) -> np.ndarray:
        """Encodes texts into normalized 1024-d float32 embeddings."""
        pass

    @abstractmethod
    def release(self):
        """Releases memory/resources."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._model = None
        self._lock = Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                logger.info(f"[GovVerify] RAM before Local BGE load: {get_process_memory_mb()} MB")
                check_safe_memory_for_model(EMBEDDING_MODEL)
                logger.info(f"[GovVerify] Loading local BGE-M3 model on CPU: {EMBEDDING_MODEL}")
                from sentence_transformers import SentenceTransformer
                try:
                    model = SentenceTransformer(
                        EMBEDDING_MODEL,
                        device='cpu',
                        model_kwargs={'low_cpu_mem_usage': True}
                    )
                    model.eval()
                except (MemoryError, RuntimeError) as mem_err:
                    logger.error(f"Failed to allocate memory for BGE-M3 model: {mem_err}")
                    raise InsufficientMemoryError(
                        f"Out of memory allocating BGE-M3 on CPU: {mem_err}"
                    )
                self._model = model
                gc.collect()
                logger.info(f"[GovVerify] RAM after Local BGE load: {get_process_memory_mb()} MB")
        return self._model

    def encode(self, texts: Union[str, List[str]], batch_size: int = 4) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, 1024), dtype='float32')

        model = self._get_model()
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
        return np.asarray(embeddings, dtype='float32')

    def release(self):
        with self._lock:
            if self._model is not None:
                logger.info("Releasing local BGE-M3 model from RAM...")
                del self._model
                self._model = None
                gc.collect()
                logger.info(f"[GovVerify] RAM after cleanup: {get_process_memory_mb()} MB")


class RemoteEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or EMBEDDING_API_URL
        self.api_key = api_key or HF_API_KEY
        self.timeout = INFERENCE_TIMEOUT_SEC

    def encode(self, texts: Union[str, List[str]], batch_size: int = 4) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, 1024), dtype='float32')

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        all_embs = []
        # Process in batches
        for i in range(0, len(texts), max(1, batch_size)):
            batch = texts[i:i + batch_size]
            payload = {
                'inputs': batch if len(batch) > 1 else batch[0],
                'options': {'wait_for_model': True, 'use_cache': True}
            }
            try:
                logger.info(f"[REMOTE_EMBEDDING] Sending {len(batch)} text(s) to BGE-M3 provider: {self.api_url[:50]}...")
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.api_url, json=payload, headers=headers)
                
                if resp.status_code != 200:
                    err_msg = f"Remote BGE-M3 inference returned status {resp.status_code}: {resp.text[:200]}"
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)

                raw = resp.json()
                # If raw is a 1D list (single text), wrap in list
                if isinstance(raw, list) and raw and isinstance(raw[0], (int, float)):
                    raw = [raw]
                elif isinstance(raw, dict) and 'embeddings' in raw:
                    raw = raw['embeddings']

                batch_arr = np.asarray(raw, dtype='float32')
                # Guarantee unit normalization for cosine similarity
                norms = np.linalg.norm(batch_arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                batch_norm = batch_arr / norms
                all_embs.append(batch_norm)

            except Exception as e:
                logger.error(f"[REMOTE_EMBEDDING_ERROR] Failed to query remote BGE-M3 provider: {e}")
                raise RuntimeError(f"Remote BGE-M3 embedding service unavailable: {e}")

        if all_embs:
            return np.vstack(all_embs).astype('float32')
        return np.empty((0, 1024), dtype='float32')

    def release(self):
        # Stateless remote client
        pass


# ==============================================================================
# NLI Providers (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
# ==============================================================================

class NLIProvider(ABC):
    @abstractmethod
    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Returns normalized probability distribution {'entailment': float, 'neutral': float, 'contradiction': float}."""
        pass

    @abstractmethod
    def release(self):
        pass


class LocalNLIProvider(NLIProvider):
    def __init__(self):
        self._nli = None
        self._lock = Lock()

    def _get_model(self):
        if self._nli is not None:
            return self._nli

        with self._lock:
            if self._nli is None:
                logger.info(f"[GovVerify] RAM before Local DeBERTa: {get_process_memory_mb()} MB")
                check_safe_memory_for_model(NLI_MODEL, required_headroom_mb=250.0)
                logger.info(f"[GovVerify] Loading local DeBERTa NLI model on CPU: {NLI_MODEL}")
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                try:
                    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        NLI_MODEL,
                        low_cpu_mem_usage=True
                    )
                    model.to('cpu')
                    model.eval()
                except (MemoryError, RuntimeError) as mem_err:
                    logger.error(f"Failed to allocate memory for DeBERTa model: {mem_err}")
                    raise InsufficientMemoryError(
                        f"Out of memory allocating DeBERTa on CPU: {mem_err}"
                    )
                self._nli = (tok, model)
                gc.collect()
                logger.info(f"[GovVerify] RAM after Local DeBERTa: {get_process_memory_mb()} MB")
        return self._nli

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        tok, model = self._get_model()
        inputs = tok(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512)
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        entail_idx = next((k for k, v in id2label.items() if 'entail' in v), 0)
        neutral_idx = next((k for k, v in id2label.items() if 'neutral' in v), 1)
        contrad_idx = next((k for k, v in id2label.items() if 'contrad' in v), 2)

        return {
            'entailment': round(float(probs[entail_idx]), 4),
            'neutral': round(float(probs[neutral_idx]), 4),
            'contradiction': round(float(probs[contrad_idx]), 4)
        }

    def release(self):
        with self._lock:
            if self._nli is not None:
                logger.info("Releasing local DeBERTa NLI model from RAM...")
                del self._nli
                self._nli = None
                gc.collect()
                logger.info(f"[GovVerify] RAM after cleanup: {get_process_memory_mb()} MB")


class RemoteNLIProvider(NLIProvider):
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or NLI_API_URL
        self.api_key = api_key or HF_API_KEY
        self.timeout = INFERENCE_TIMEOUT_SEC

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        # Standard zero-shot / sentence-pair inference format
        payload = {
            'inputs': f"{premise} </s></s> {hypothesis}",
            'options': {'wait_for_model': True, 'use_cache': True}
        }

        try:
            logger.info(f"[REMOTE_NLI] Sending premise & claim pair to DeBERTa provider: {self.api_url[:50]}...")
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.api_url, json=payload, headers=headers)

            if resp.status_code != 200:
                # Try structured pair payload if format 1 failed
                alt_payload = {
                    'inputs': {'text': premise, 'text_pair': hypothesis},
                    'options': {'wait_for_model': True}
                }
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.api_url, json=alt_payload, headers=headers)

            if resp.status_code != 200:
                err_msg = f"Remote DeBERTa NLI inference returned status {resp.status_code}: {resp.text[:200]}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            raw = resp.json()
            # Parse HuggingFace classification score lists [[{"label": "...", "score": 0.9}]]
            results_list = raw[0] if (isinstance(raw, list) and raw and isinstance(raw[0], list)) else (raw if isinstance(raw, list) else [])
            
            p_entail = 0.33
            p_neutral = 0.34
            p_contrad = 0.33

            for item in results_list:
                if isinstance(item, dict):
                    lbl = str(item.get('label', '')).lower()
                    score = float(item.get('score', 0.0))
                    if 'entail' in lbl:
                        p_entail = score
                    elif 'neutral' in lbl:
                        p_neutral = score
                    elif 'contrad' in lbl:
                        p_contrad = score

            return {
                'entailment': round(p_entail, 4),
                'neutral': round(p_neutral, 4),
                'contradiction': round(p_contrad, 4)
            }

        except Exception as e:
            logger.error(f"[REMOTE_NLI_ERROR] Failed to query remote DeBERTa provider: {e}")
            raise RuntimeError(f"Remote DeBERTa NLI service unavailable: {e}")

    def release(self):
        pass


# ==============================================================================
# Global Singleton Accessors with Auto-Detection
# ==============================================================================

_local_embedding_provider = LocalEmbeddingProvider()
_remote_embedding_provider = RemoteEmbeddingProvider()
_local_nli_provider = LocalNLIProvider()
_remote_nli_provider = RemoteNLIProvider()


def get_embedding_provider() -> EmbeddingProvider:
    provider = INFERENCE_PROVIDER.lower()
    if provider in ['remote', 'huggingface', 'hf', 'api']:
        return _remote_embedding_provider
    elif provider == 'local':
        return _local_embedding_provider
    
    # Auto mode:
    # If HF_API_KEY is configured OR running in memory-constrained environment (<=600MB like Render Free)
    if HF_API_KEY or get_container_memory_limit_mb() <= 600.0:
        logger.info("[PROVIDER_ROUTER] Routing BGE-M3 to Remote Provider (Render Free/Cloud mode)")
        return _remote_embedding_provider
    
    return _local_embedding_provider


def get_nli_provider() -> NLIProvider:
    provider = INFERENCE_PROVIDER.lower()
    if provider in ['remote', 'huggingface', 'hf', 'api']:
        return _remote_nli_provider
    elif provider == 'local':
        return _local_nli_provider
    
    # Auto mode:
    if HF_API_KEY or get_container_memory_limit_mb() <= 600.0:
        logger.info("[PROVIDER_ROUTER] Routing DeBERTa NLI to Remote Provider (Render Free/Cloud mode)")
        return _remote_nli_provider
    
    return _local_nli_provider
