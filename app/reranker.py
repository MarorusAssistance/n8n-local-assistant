from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import importlib
import logging
import threading

from .config import settings


class Reranker:
    """Singleton-style reranker wrapper for FlagEmbedding models."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Optional[Any] = None
        self._model_name: Optional[str] = None
        self._supports_compression = False
        self._logger = logging.getLogger("n8n-assistant")

    def enabled(self) -> bool:
        return bool(settings.ENABLE_RERANK)

    def load(self) -> bool:
        if not settings.ENABLE_RERANK:
            return False
        with self._lock:
            if self._model is not None:
                return True
            return self._load_with_fallback()

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not settings.ENABLE_RERANK:
            return list(candidates)
        if not candidates:
            return []
        if not self.load():
            return list(candidates)

        trimmed_pairs, filtered_candidates = self._build_pairs(query, candidates)
        if not trimmed_pairs:
            return list(candidates)

        try:
            scores = self._compute_scores(trimmed_pairs)
        except Exception:
            self._logger.exception(
                "reranker scoring failed: id=%s model=%s",
                request_id or "-",
                self._model_name or settings.RERANK_MODEL,
            )
            return list(candidates)

        scored: List[Tuple[Dict[str, Any], float]] = []
        for chunk, score in zip(filtered_candidates, scores):
            chunk = dict(chunk)
            chunk["rerank_score"] = float(score)
            scored.append((chunk, float(score)))

        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [chunk for chunk, _ in scored]

        if top_k is not None and top_k > 0:
            return ranked[:top_k]
        return ranked

    def _build_pairs(
        self, query: str, candidates: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        max_chars = settings.RERANK_MAX_CHARS
        pairs: List[Tuple[str, str]] = []
        filtered: List[Dict[str, Any]] = []
        for chunk in candidates:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            trimmed = self._truncate(text, max_chars=max_chars)
            pairs.append((query, trimmed))
            filtered.append(chunk)
        return pairs, filtered

    def _truncate(self, text: str, max_chars: int) -> str:
        if not max_chars or max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip()

    def _compute_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if self._model is None:
            return []
        kwargs = self._compress_kwargs()
        scores = self._model.compute_score(pairs, **kwargs)
        return [float(score) for score in scores]

    def _compress_kwargs(self) -> Dict[str, Any]:
        if not self._supports_compression:
            return {}
        kwargs: Dict[str, Any] = {}
        cutoff_layers = _parse_int_list(settings.RERANK_CUTOFF_LAYERS)
        compress_layers = _parse_int_list(settings.RERANK_COMPRESS_LAYERS)
        if cutoff_layers:
            kwargs["cutoff_layers"] = cutoff_layers
        if compress_layers:
            kwargs["compress_layers"] = compress_layers
        if settings.RERANK_COMPRESS_RATIO:
            kwargs["compress_ratio"] = settings.RERANK_COMPRESS_RATIO
        return kwargs

    def _load_with_fallback(self) -> bool:
        model_name = settings.RERANK_MODEL
        try:
            self._model = self._load_model(model_name)
            self._model_name = model_name
            self._logger.info("reranker ready: model=%s", model_name)
            return True
        except Exception as exc:
            self._logger.warning("reranker load failed: model=%s error=%s", model_name, exc)
            fallback = settings.RERANK_FALLBACK_MODEL or ""
            if not fallback or fallback == model_name:
                return False
            try:
                self._model = self._load_model(fallback, force_cpu=True)
                self._model_name = fallback
                self._logger.info("reranker fallback ready: model=%s", fallback)
                return True
            except Exception as exc2:
                self._logger.warning(
                    "reranker fallback failed: model=%s error=%s", fallback, exc2
                )
                return False

    def _load_model(self, model_name: str, force_cpu: bool = False) -> Any:
        model_name = model_name.strip()
        use_fp16 = bool(settings.RERANK_USE_FP16)
        devices = _resolve_devices(force_cpu=force_cpu)
        batch_size = settings.RERANK_BATCH_SIZE or None
        model_kwargs: Dict[str, Any] = {
            "use_fp16": use_fp16,
            "devices": devices,
        }
        # Some FlagEmbedding versions break if batch_size is passed as None.
        if batch_size is not None:
            model_kwargs["batch_size"] = batch_size

        if "bge-reranker-v2.5" in model_name and "lightweight" in model_name:
            _ensure_lightweight_compatible()
            from FlagEmbedding import LightWeightFlagLLMReranker

            self._supports_compression = True
            return LightWeightFlagLLMReranker(
                model_name,
                **model_kwargs,
            )

        from FlagEmbedding import FlagReranker

        self._supports_compression = False
        return FlagReranker(
            model_name,
            **model_kwargs,
        )



def _ensure_lightweight_compatible() -> None:
    """Guard against known transformers/FlagEmbedding incompatibilities."""
    try:
        gemma2 = importlib.import_module("transformers.models.gemma2.modeling_gemma2")
    except Exception as exc:
        raise RuntimeError(
            "Lightweight reranker requires transformers Gemma2 modules"
        ) from exc
    if not hasattr(gemma2, "GEMMA2_START_DOCSTRING"):
        raise RuntimeError(
            "Installed transformers is incompatible with "
            "BAAI/bge-reranker-v2.5-gemma2-lightweight. "
            "Pin a compatible 4.x release or use fallback model."
        )
def _resolve_devices(force_cpu: bool = False) -> Optional[List[str]]:
    if force_cpu:
        return ["cpu"]
    device = (settings.RERANK_DEVICE or "").strip().lower()
    if device:
        return [device]
    try:
        import torch

        if torch.cuda.is_available():
            return ["cuda:0"]
    except Exception:
        return ["cpu"]
    return ["cpu"]


def _parse_int_list(value: Optional[str]) -> List[int]:
    if not value:
        return []
    parts = [item.strip() for item in value.split(",") if item.strip()]
    result: List[int] = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            continue
    return result


reranker = Reranker()
