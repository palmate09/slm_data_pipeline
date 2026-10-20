"""Domain gate: two-layer ML-topic filter.

Layer 1: keyword/rule-based prefilter (fast, cheap).
Layer 2: embedding-based cosine similarity vs an ML prototype set (accurate).

Documents must pass BOTH layers to be kept. The keyword layer rejects obviously
off-topic documents before the (more expensive) embedding model runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # optional heavy dependency
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    _HAS_ST = True
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]
    cos_sim = None  # type: ignore[assignment]
    _HAS_ST = False


@dataclass
class GateResult:
    keep: bool
    reason: str
    keyword_hits: int
    keyword_distinct: int = 0
    topic_similarity: float | None = None
    layer: str = "none"  # "keyword" | "embedding" | "both"


class DomainGate:
    def __init__(
        self,
        *,
        topic: str = "machine_learning",
        keywords: Iterable[str] | None = None,
        keyword_min_hits: int = 2,
        embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2",
        prototype_set_path: str | Path | None = None,
        topic_similarity_threshold: float = 0.6,
        enable_embedding: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.topic = topic
        self.keywords = [k.lower() for k in (keywords or [])]
        self.keyword_min_hits = max(1, keyword_min_hits)
        self.embedding_model_name = embedding_model
        self.prototype_set_path = Path(prototype_set_path) if prototype_set_path else None
        self.topic_similarity_threshold = topic_similarity_threshold
        self.enable_embedding = enable_embedding and _HAS_ST
        self.batch_size = batch_size

        self._model = None
        self._prototype_embeddings: np.ndarray | None = None
        self._prototype_texts: list[str] = []

    # ------------------------------------------------------------------ public
    def gate(self, document) -> GateResult:
        """Return whether the document is on-topic and why."""
        text = getattr(document, "text", "") or ""
        if not text.strip():
            return GateResult(
                keep=False, reason="empty text", keyword_hits=0, layer="none",
            )

        # ---- Layer 1: occurrence-weighted keyword prefilter
        title = getattr(document, "title", "") or ""
        headings = getattr(document, "headings", None) or []
        keyword_hits, keyword_distinct = self._count_keyword_hits(text, title, headings)
        if keyword_hits < self.keyword_min_hits:
            return GateResult(
                keep=False,
                reason=f"too few keyword hits ({keyword_hits} < {self.keyword_min_hits})",
                keyword_hits=keyword_hits, keyword_distinct=keyword_distinct,
                layer="keyword",
            )

        # ---- Layer 2: embedding classifier
        if not self.enable_embedding:
            return GateResult(
                keep=True, reason="keyword pass (embedding disabled)",
                keyword_hits=keyword_hits, keyword_distinct=keyword_distinct,
                layer="keyword",
            )

        try:
            if self._prototype_embeddings is None:
                self._load_prototypes()
        except Exception:  # noqa: BLE001 - model download/runtime failure
            self._prototype_embeddings = None

        if self._prototype_embeddings is None or len(self._prototype_embeddings) == 0:
            # No prototype set: fall back to keyword decision.
            return GateResult(
                keep=True, reason="keyword pass (no prototype set)",
                keyword_hits=keyword_hits, keyword_distinct=keyword_distinct,
                layer="keyword",
            )

        # Include the title and headings in the embedding input: short
        # tutorial pages often have little body text but a very clear title.
        embedding_text = "\n".join(
            part for part in [title, " ".join(headings), text] if part
        )
        try:
            similarity = self._max_similarity(embedding_text)
        except Exception:  # noqa: BLE001 - model unavailable at runtime
            return GateResult(
                keep=True, reason="keyword pass (embedding unavailable)",
                keyword_hits=keyword_hits, keyword_distinct=keyword_distinct,
                layer="keyword",
            )
        if similarity >= self.topic_similarity_threshold:
            return GateResult(
                keep=True, reason="on topic",
                keyword_hits=keyword_hits, keyword_distinct=keyword_distinct,
                topic_similarity=float(similarity), layer="embedding",
            )
        return GateResult(
            keep=False,
            reason=f"below topic threshold ({similarity:.3f} < {self.topic_similarity_threshold})",
            keyword_hits=keyword_hits, keyword_distinct=keyword_distinct,
            topic_similarity=float(similarity), layer="embedding",
        )

    # ------------------------------------------------------------------ keyword
    def _count_keyword_hits(self, text: str, title: str = "", headings: Iterable[str] = ()) -> tuple[int, int]:
        """Count occurrences, weighting title/headings more than body text.

        Counting distinct keywords (the old behaviour) made a long, genuinely
        on-topic article fail when it only used one phrase; counting every
        occurrence can make a passing mention look strong.  Weighted
        occurrences with per-keyword saturation is a compromise that favours
        documents which actually discuss the topic.
        """
        weighted = 0
        distinct = 0
        title_l = (title or "").lower()
        headings_l = " \n ".join(headings or []).lower()
        body_l = (text or "").lower()
        for kw in self.keywords:
            kw = kw.strip().lower()
            if not kw:
                continue
            body_count = body_l.count(kw)
            title_count = title_l.count(kw)
            heading_count = headings_l.count(kw)
            if body_count or title_count or heading_count:
                distinct += 1
            weighted += min(body_count, 20) * 1
            weighted += min(title_count, 5) * 3
            weighted += min(heading_count, 10) * 2
        return int(weighted), int(distinct)

    # ------------------------------------------------------------------ embedding
    def _load_prototypes(self) -> None:
        texts: list[str] = []
        if self.prototype_set_path and self.prototype_set_path.exists():
            try:
                texts = [
                    line.strip()
                    for line in self.prototype_set_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception:  # noqa: BLE001
                texts = []
        self._prototype_texts = texts

        if not self.enable_embedding or not texts:
            self._prototype_embeddings = None
            return

        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, batch_size=self.batch_size,
            show_progress_bar=False,
        )
        self._prototype_embeddings = np.asarray(embeddings, dtype="float32")

    def _max_similarity(self, text: str) -> float:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        # Chunk long documents so embedding stays cheap & meaningful.
        chunks = self._chunk_text(text, max_chars=512)
        if not chunks:
            return 0.0
        doc_emb = self._model.encode(
            chunks, convert_to_numpy=True, batch_size=self.batch_size,
            show_progress_bar=False,
        )
        doc_emb = np.asarray(doc_emb, dtype="float32")
        sim = cos_sim(doc_emb, self._prototype_embeddings)  # (n_chunks, n_proto)
        sim = np.asarray(sim, dtype="float32")
        return float(np.max(sim))

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 512) -> list[str]:
        # Split on sentence boundaries, then pack into chunks.
        parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        chunks: list[str] = []
        cur = ""
        for part in parts:
            if len(cur) + len(part) + 1 <= max_chars:
                cur = (cur + " " + part).strip() if cur else part
            else:
                if cur:
                    chunks.append(cur)
                cur = part if len(part) <= max_chars else part[:max_chars]
        if cur:
            chunks.append(cur)
        if not chunks:
            chunks = [text[:max_chars]]
        return chunks


def build_default_ml_keywords() -> list[str]:
    """Canonical ML keyword list used when the config provides none."""
    return [
        "machine learning", "deep learning", "neural network", "gradient descent",
        "backpropagation", "transformer", "attention mechanism", "convolution",
        "convolutional", "recurrent", "lstm", "gru", "overfitting", "underfitting",
        "loss function", "cross entropy", "mean squared error", "optimizer",
        "stochastic gradient", "adam", "sgd", "learning rate", "batch size",
        "epoch", "validation set", "test set", "training set", "bias", "variance",
        "regularization", "dropout", "batch normalization", "activation",
        "relu", "sigmoid", "softmax", "pytorch", "tensorflow", "keras",
        "scikit-learn", "scikit learn", "random forest", "gradient boosting",
        "xgboost", "support vector", "svm", "k-means", "clustering", "pca",
        "principal component", "regression", "classification", "recommendation",
        "computer vision", "natural language processing", "nlp", "reinforcement",
        "q-learning", "policy", "reward", "model", "inference", "training",
        "supervised", "unsupervised", "semi-supervised", "embedding", "token",
        "tokenizer", "language model", "transformer architecture",
    ]
