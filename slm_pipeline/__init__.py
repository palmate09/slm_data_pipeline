"""SLM Data Collection & Cleaning Pipeline.

An end-to-end, topic-specific data curation pipeline that turns a list of URLs
into a tokenization-ready, domain-filtered dataset (JSONL/Parquet with a `text`
field) for training a Small Language Model (SLM) that knows *only* that subject.

Designed after the NVIDIA NeMo Curator pattern (DocumentDataset, Sequential
modifiers, score filters, exact + fuzzy deduplication) but dependency-light so it
runs on CPU out of the box and optionally accelerates with Ray/GPU backends.
"""

from .config import PipelineConfig, load_config
from .dataset import DocumentDataset
from .models import Block, Document, new_id, now_iso
from .pipeline import run_pipeline
from .stage_f import Chunk

__all__ = [
    "Block",
    "Chunk",
    "Document",
    "DocumentDataset",
    "PipelineConfig",
    "load_config",
    "new_id",
    "now_iso",
    "run_pipeline",
]

__version__ = "0.2.0"
