"""Curriculum ordering: sort a corpus from base knowledge to advanced.

The corpus is a mixture of introductory pages (what is machine learning),
reference articles (backpropagation, transformers) and practitioner READMEs
(distributed training, fine-tuning).  For a small corpus that is read in one
or two passes, the order in which the material is presented matters: a model
(or a human) benefits from meeting the foundations before the specialised
machinery built on top of them.

The ordering is computed over the *whole* final corpus at once, so documents
that were extracted in an earlier run and documents extracted in the newest
run are ranked together instead of separately.

Difficulty is estimated from signals that are cheap, explainable and
deterministic:

* ``terms``    - weighted density of technical vocabulary
* ``math``     - density of mathematical notation
* ``code``     - density of code samples
* ``structure``- heading depth and section count
* ``cue``      - explicit cues in the title and opening ("getting started",
  "advanced", "from scratch", ...)
* ``vocabulary``- lexical rarity proxy (long words, long sentences)

The score is stored on every document and chunk, and the output files are
written in that order.  Nothing here removes the ability to shuffle: the
rank is metadata, and the chunk ids/hashes are unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..models import Document

__all__ = [
    "DifficultySignals",
    "LEVELS",
    "difficulty_score",
    "level_for_score",
    "rank_documents",
    "sort_documents",
]

# Curriculum levels, easiest first.
LEVELS = ("foundational", "intermediate", "advanced", "expert")

# Technical vocabulary with a difficulty weight in [0, 1].  The weights are
# deliberately coarse: they express "how specialised is this term", not a
# claim about the exact curriculum of any course.
_TERM_WEIGHTS: dict[str, float] = {
    # --- foundations -------------------------------------------------
    "machine learning": 0.05,
    "artificial intelligence": 0.05,
    "neural network": 0.10,
    "artificial neural network": 0.12,
    "deep learning": 0.22,
    "training data": 0.10,
    "dataset": 0.10,
    "classification": 0.12,
    "regression": 0.15,
    "supervised learning": 0.15,
    "unsupervised learning": 0.18,
    "self-supervised": 0.40,
    "overfitting": 0.20,
    "activation function": 0.22,
    "loss function": 0.24,
    "gradient": 0.26,
    "feature": 0.20,
    "linear regression": 0.20,
    "logistic regression": 0.25,
    "naive bayes": 0.28,
    "decision tree": 0.25,
    "random forest": 0.30,
    "k-means": 0.30,
    "principal component analysis": 0.35,
    "dimensionality reduction": 0.32,
    "cross validation": 0.28,
    "gradient boosting": 0.35,
    "support vector machine": 0.35,
    "ensemble learning": 0.35,
    "transfer learning": 0.35,
    "feature engineering": 0.30,
    "computer vision": 0.30,
    "natural language processing": 0.35,
    "reinforcement learning": 0.35,
    "convolutional neural network": 0.35,
    "recurrent neural network": 0.36,
    "autoencoder": 0.40,
    "generative adversarial network": 0.55,
    "variational autoencoder": 0.60,
    "diffusion model": 0.62,
    # --- intermediate ------------------------------------------------
    "gradient descent": 0.32,
    "stochastic gradient descent": 0.35,
    "backpropagation": 0.36,
    "chain rule": 0.30,
    "convolution": 0.34,
    "regularization": 0.38,
    "batch normalization": 0.42,
    "layer normalization": 0.50,
    "dropout": 0.30,
    "momentum": 0.30,
    "learning rate": 0.28,
    "hyperparameter": 0.46,
    "cross-entropy": 0.40,
    "softmax": 0.40,
    "embedding": 0.42,
    "word embedding": 0.45,
    "tokenizer": 0.44,
    "subword": 0.52,
    "byte pair encoding": 0.52,
    "wordpiece": 0.50,
    "perplexity": 0.46,
    "inference": 0.48,
    "language model": 0.45,
    "autoregressive": 0.58,
    "attention": 0.54,
    "attention mechanism": 0.50,
    "multi-head attention": 0.62,
    "self-attention": 0.68,
    "cross-attention": 0.60,
    "positional encoding": 0.55,
    "rotary positional embedding": 0.68,
    "alibi": 0.62,
    "transformer": 0.58,
    "encoder decoder": 0.40,
    "sequence-to-sequence": 0.42,
    "knowledge distillation": 0.58,
    "pre-training": 0.56,
    "fine-tuning": 0.58,
    "instruction tuning": 0.60,
    "context window": 0.50,
    "kv cache": 0.65,
    "paged attention": 0.70,
    "meta-learning": 0.45,
    "federated learning": 0.50,
    "continual learning": 0.50,
    "multi-task learning": 0.45,
    "curriculum learning": 0.50,
    "active learning": 0.42,
    "hyperparameter optimization": 0.45,
    "interpretability": 0.55,
    "explainability": 0.50,
    "adversarial": 0.50,
    "robustness": 0.45,
    "fairness": 0.40,
    "anomaly detection": 0.30,
    "recommender system": 0.35,
    "collaborative filtering": 0.40,
    "matrix factorization": 0.40,
    "graph neural network": 0.55,
    "retrieval-augmented generation": 0.62,
    "vector database": 0.55,
    "embedding model": 0.50,
    "re-ranking": 0.55,
    "dense retrieval": 0.62,
    "bm25": 0.55,
    # --- advanced ----------------------------------------------------
    "large language model": 0.60,
    "large language models": 0.60,
    "foundation model": 0.60,
    "mixture of experts": 0.80,
    "expert parallelism": 0.78,
    "pipeline parallelism": 0.75,
    "data parallelism": 0.70,
    "tensor parallelism": 0.76,
    "sequence parallelism": 0.76,
    "model parallel": 0.72,
    "zero redundancy": 0.72,
    "fully sharded data parallel": 0.72,
    "fsdp": 0.68,
    "megatron": 0.78,
    "deepspeed": 0.76,
    "fairscale": 0.70,
    "nccl": 0.75,
    "all-reduce": 0.70,
    "collective communication": 0.68,
    "sharding": 0.70,
    "gradient accumulation": 0.72,
    "gradient checkpointing": 0.74,
    "activation checkpointing": 0.74,
    "mixed precision": 0.68,
    "bf16": 0.62,
    "fp16": 0.58,
    "quantization": 0.68,
    "int8": 0.60,
    "int4": 0.60,
    "nf4": 0.65,
    "gptq": 0.68,
    "awq": 0.68,
    "gguf": 0.65,
    "ggml": 0.70,
    "llama.cpp": 0.72,
    "vllm": 0.70,
    "sglang": 0.72,
    "tensorrt": 0.70,
    "triton": 0.72,
    "triton kernel": 0.78,
    "cutlass": 0.75,
    "flashattention": 0.84,
    "flash-decoding": 0.78,
    "speculative decoding": 0.78,
    "parameter-efficient": 0.76,
    "lora": 0.74,
    "qlora": 0.76,
    "peft": 0.76,
    "prefix tuning": 0.72,
    "prompt tuning": 0.70,
    "8-bit": 0.60,
    "4-bit": 0.60,
    "zero-shot": 0.66,
    "few-shot": 0.55,
    "in-context learning": 0.60,
    "chain of thought": 0.60,
    "scaling law": 0.76,
    "compute-optimal": 0.75,
    "pretraining data": 0.60,
    "deduplication": 0.50,
    "minhash": 0.60,
    "synthetic data": 0.55,
    "reinforcement learning from human feedback": 0.82,
    "rlhf": 0.82,
    "direct preference optimization": 0.82,
    "dpo": 0.82,
    "grpo": 0.75,
    "reward model": 0.70,
    "preference optimization": 0.75,
    "ppo": 0.62,
    "actor critic": 0.55,
    "policy gradient": 0.50,
    "q-learning": 0.45,
    "monte carlo tree search": 0.58,
    "cifar": 0.30,
    "imagenet": 0.30,
    "mnist": 0.25,
    "mmlu": 0.55,
    "gsm8k": 0.55,
    "humaneval": 0.60,
    "pass@k": 0.60,
    "bleu": 0.45,
    "rouge": 0.45,
    "bits per byte": 0.55,
    "nucleus sampling": 0.55,
    "beam search": 0.50,
    "tensorboard": 0.40,
    "experiment tracking": 0.35,
    "safetensors": 0.60,
    "chat template": 0.50,
}

# Title/lead cues that signal the intended level of a page.
_FOUNDATIONAL_CUES = (
    "introduction", "getting started", "what is", "overview", "history",
    "basics", "fundamental", "primer", "beginner", "explained", "101",
)
_ADVANCED_CUES = (
    "advanced", "deep dive", "from scratch", "mastering", "scaling",
    "distributed", "in production", "handbook", "cookbook", "architecture of",
)

_MATH_RE = re.compile(r"[∂∇∑∫√≈≤≥≠⊗∈]|\b\w+\s*=\s*[-\w.]")
_CODE_HINT_RE = re.compile(
    r"^\s*(?:```|from |import |def |class |pip |git |python |bash |sh |"
    r"\$ |sudo |npm |cargo |make |curl )"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


@dataclass(frozen=True)
class DifficultySignals:
    """Explainable breakdown of one document's difficulty score."""

    score: float
    level: str
    terms: float = 0.0
    math: float = 0.0
    code: float = 0.0
    structure: float = 0.0
    cue: float = 0.0
    vocabulary: float = 0.0
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "level": self.level,
            "signals": {
                "terms": round(self.terms, 4),
                "math": round(self.math, 4),
                "code": round(self.code, 4),
                "structure": round(self.structure, 4),
                "cue": round(self.cue, 4),
                "vocabulary": round(self.vocabulary, 4),
            },
            "matched_terms": self.matched_terms[:12],
        }


def level_for_score(score: float) -> str:
    """Map a 0-1 difficulty score onto a curriculum level."""
    if score < 0.30:
        return "foundational"
    if score < 0.55:
        return "intermediate"
    if score < 0.78:
        return "advanced"
    return "expert"


def _text_of(document: Document) -> str:
    return getattr(document, "text", "") or ""


def _split_code(text: str) -> tuple[str, str]:
    """Return (prose, code) parts of a rendered document."""
    prose_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        (code_lines if in_code else prose_lines).append(line)
    return "\n".join(prose_lines), "\n".join(code_lines)


def _term_signal(prose: str, words: int) -> tuple[float, list[str]]:
    """Difficulty of the vocabulary, weighted by the most specialised terms.

    A general article mentions a little of everything, while a specialist
    page concentrates on a few hard terms, so the score combines the average
    of the hardest matched terms with the single hardest match.
    """
    lowered = prose.lower()
    matched: list[tuple[float, int, str]] = []
    for term, weight in _TERM_WEIGHTS.items():
        count = lowered.count(term)
        if count:
            matched.append((weight, count, term))
    if not matched:
        return 0.05, []
    matched.sort(key=lambda item: (-item[0], item[2]))
    top = matched[:5]
    average = sum(weight for weight, _, _ in top) / len(top)
    peak = matched[0][0]
    density = sum(count for _, count, _ in matched) / max(1.0, words / 200.0)
    score = 0.5 * average + 0.3 * peak + 0.2 * min(1.0, density)
    terms = [term for _, _, term in matched]
    return max(0.0, min(1.0, score)), terms


def _topic_signal(document: Document) -> float:
    """Difficulty of the page's subject, taken from its title.

    The title states what the page is *about*; for a curriculum that is a
    stronger signal than the body vocabulary.
    """
    title = (getattr(document, "title", "") or "").lower()
    if not title:
        return 0.10
    weights = [weight for term, weight in _TERM_WEIGHTS.items() if term in title]
    return max(weights) if weights else 0.10


def _math_signal(prose: str) -> float:
    lines = [line for line in prose.splitlines() if line.strip()]
    if not lines:
        return 0.0
    math_lines = sum(1 for line in lines if _MATH_RE.search(line))
    return max(0.0, min(1.0, 3.0 * math_lines / len(lines)))


def _code_signal(text: str, prose: str) -> float:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    code_lines = sum(1 for line in lines if _CODE_HINT_RE.match(line))
    return max(0.0, min(1.0, 2.0 * code_lines / len(lines)))


def _structure_signal(document: Document, prose: str) -> float:
    headings = [b for b in getattr(document, "blocks", []) or [] if b.type == "heading"]
    if not headings:
        headings_text = re.findall(r"(?m)^#{1,6}\s+.+$", prose or "")
        depth = 0.0
        count = len(headings_text)
    else:
        levels = [b.level or 2 for b in headings]
        depth = (sum(levels) / len(levels) - 1.0) / 4.0
        count = len(headings)
    return max(0.0, min(1.0, 0.6 * max(0.0, depth) + 0.4 * min(1.0, count / 25.0)))


def _cue_signal(document: Document, prose: str) -> float:
    """Explicit level cues in the title and the opening paragraph."""
    title = (getattr(document, "title", "") or "").lower()
    opening = " ".join(prose.split()[:120]).lower()
    head = f"{title} {opening}"
    score = 0.5
    for cue in _FOUNDATIONAL_CUES:
        if re.search(rf"\b{re.escape(cue)}\b", head):
            score -= 0.18
    for cue in _ADVANCED_CUES:
        if re.search(rf"\b{re.escape(cue)}\b", head):
            score += 0.18
    return max(0.0, min(1.0, score))


def _vocabulary_signal(prose: str) -> float:
    words = _WORD_RE.findall(prose)
    if not words:
        return 0.0
    long_words = sum(1 for word in words if len(word) >= 8)
    average = sum(len(word) for word in words) / len(words)
    return max(0.0, min(1.0, 0.6 * (long_words / len(words)) + 0.4 * ((average - 4.0) / 5.0)))


def difficulty_score(document: Document) -> DifficultySignals:
    """Score one document on a 0-1 difficulty scale."""
    text = _text_of(document)
    prose, _ = _split_code(text)
    words = len(prose.split()) or 1

    terms, matched = _term_signal(prose, words)
    topic = _topic_signal(document)
    math = _math_signal(prose)
    code = _code_signal(text, prose)
    structure = _structure_signal(document, prose)
    cue = _cue_signal(document, prose)
    vocabulary = _vocabulary_signal(prose)

    score = (
        0.25 * topic
        + 0.30 * terms
        + 0.15 * math
        + 0.10 * code
        + 0.08 * structure
        + 0.07 * cue
        + 0.05 * vocabulary
    )
    score = max(0.0, min(1.0, score))
    return DifficultySignals(
        score=score,
        level=level_for_score(score),
        terms=terms,
        math=math,
        code=code,
        structure=structure,
        cue=cue,
        vocabulary=vocabulary,
        matched_terms=matched,
    )


def _normalise(scores: list[float]) -> list[float]:
    """Stretch raw scores across the corpus so the levels are meaningful.

    Absolute thresholds only make sense for a whole corpus, so the score is
    rescaled between the 10th and 90th percentile of the corpus.  The
    ordering is unchanged; only the spacing moves.
    """
    if not scores:
        return []
    ordered = sorted(scores)
    last = len(ordered) - 1

    def quantile(p: float) -> float:
        index = min(last, max(0, int(round(p * last))))
        return ordered[index]

    low, high = quantile(0.10), quantile(0.90)
    if high - low < 1e-9:
        return [0.5 for _ in scores]
    return [max(0.0, min(1.0, (value - low) / (high - low))) for value in scores]


def sort_documents(documents: Iterable[Document]) -> list[Document]:
    """Return the documents ordered from base knowledge to advanced.

    The sort is stable and deterministic: difficulty first, then title and id
    as tie-breakers, so repeated runs produce the same curriculum.
    """
    scored = []
    for doc in documents:
        signals = difficulty_score(doc)
        scored.append((signals.score, (getattr(doc, "title", "") or "").lower(), getattr(doc, "id", ""), doc, signals))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in scored]


def rank_documents(documents: Iterable[Document]) -> list[Document]:
    """Order documents and stamp the difficulty metadata on each one.

    Every document receives:

    * ``difficulty``       - the corpus-relative 0-1 score
    * ``curriculum_rank``  - 1-based position in the curriculum
    * ``curriculum_level`` - foundational / intermediate / advanced / expert
    * ``metadata['curriculum']`` - the full signal breakdown

    The whole corpus is scored in one pass, so documents from earlier runs
    and newly extracted documents are ranked against each other.
    """
    docs = list(documents)
    if not docs:
        return []
    signals = {id(doc): difficulty_score(doc) for doc in docs}
    normalised = _normalise([signals[id(doc)].score for doc in docs])

    pairs = []
    for doc, norm in zip(docs, normalised):
        pairs.append((norm, (getattr(doc, "title", "") or "").lower(), getattr(doc, "id", ""), doc))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))

    for index, (norm, _title, _id, doc) in enumerate(pairs, start=1):
        signal = signals[id(doc)]
        doc.difficulty = round(norm, 4)
        doc.curriculum_rank = index
        doc.curriculum_level = level_for_score(norm)
        doc.metadata["curriculum"] = {
            **signal.to_dict(),
            "score": round(norm, 4),
            "raw_score": round(signal.score, 4),
            "level": level_for_score(norm),
        }
    return [item[3] for item in pairs]
