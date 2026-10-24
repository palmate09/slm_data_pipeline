"""Topic configuration loading & validation.

A `topic_config.yaml` drives the whole pipeline. Swap this file (keywords,
prototype set, thresholds, source adapters) to retarget the pipeline at a new
subject; the stage interfaces stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    # ---- topic ----
    topic: str
    display_name: str | None = None

    # ---- ingestion ----
    input_urls_file: str = "data/urls/ml_urls.txt"
    source_type: str = "urls"  # urls | sitemap | local | jsonl
    sitemap_url: str | None = None
    local_input_dir: str | None = None
    jsonl_input_file: str | None = None
    follow_sitemaps: bool = False
    max_pages: int | None = None

    respect_robots_txt: bool = True
    request_delay_seconds: float = 1.0
    max_retries: int = 3
    request_timeout_seconds: int = 30
    user_agent: str = "slm-data-pipeline/0.2 (contact: example@example.com)"
    download_dir: str = "data/downloads"

    # ---- extraction (Stage A) ----
    extractor: str = "auto"  # auto | wikipedia | trafilatura | readability | justext | pdf | text
    min_extraction_confidence: float = 0.25
    content_profile: str = "prose"  # prose | technical | code
    preserve_code: bool = True
    preserve_tables: bool = False
    preserve_math: bool = False
    math_policy: str = "text"  # text | latex | drop
    table_policy: str = "drop"  # drop | markdown | csv
    code_policy: str = "keep"  # keep | drop
    caption_policy: str = "drop"  # drop | keep
    reference_policy: str = "drop"  # drop | meta
    heading_markup: bool = True

    # ---- domain gate (Stage B) ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    prototype_set_path: str | None = "configs/ml_prototypes.txt"
    topic_similarity_threshold: float = 0.6
    keyword_min_hits: int = 2
    keywords: list[str] = field(default_factory=list)
    enable_embedding: bool = True

    # ---- browser rendering fallback (Stage A) ----
    # JS-only pages (SPA app shells) yield no server-rendered text.  When
    # enabled, the pipeline re-renders those pages with a headless system
    # browser and extracts from the DOM.  Requires chrome/chromium on PATH
    # (or chrome_binary set); failures fall back to the plain extraction.
    browser_fallback: bool = False
    chrome_binary: str | None = None
    render_timeout_seconds: float = 45.0
    render_min_words: int = 40
    render_max_pages: int = 40

    # ---- cleaning (Stage C) ----
    fix_unicode: bool = True
    unify_quotes: bool = True
    redact_pii: bool = True
    pii_entities: list[str] = field(
        default_factory=lambda: [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "US_SSN",
            "CREDIT_CARD",
            "IBAN_CODE",
        ]
    )
    pii_person: bool = False
    use_presidio: bool = False
    language: str = "en"
    min_language_confidence: float = 0.9
    remove_references: bool = True
    remove_captions: bool = True
    remove_math: bool = True
    repair_spacing: bool = True
    boilerplate_fingerprint: bool = True
    boilerplate_min_docs: int = 3

    # ---- quality filtering (Stage D) ----
    min_word_count: int = 50
    max_word_count: int = 100_000
    min_quality_score: float = 0.55
    max_perplexity: float | None = 1000.0
    perplexity_model_path: str | None = None
    min_fluency: float = 0.0
    require_lead: bool = False
    allow_leading_heading: bool = False
    min_avg_paragraph_words: float = 0.0
    max_heading_ratio: float = 1.0
    min_alpha_ratio: float = 0.6
    max_markup_ratio: float = 0.005
    max_reference_ratio: float = 0.01
    max_fragment_ratio: float = 0.15
    max_formula_ratio: float = 0.02
    max_code_ratio: float = 0.5
    max_duplicate_line_ratio: float = 0.05
    max_url_ratio: float = 0.02
    max_nonprintable_ratio: float = 0.01
    min_paragraphs: int = 2
    min_sentences: int = 5
    ngram_filters: list[dict] = field(
        default_factory=lambda: [
            {"n": 2, "max_ratio": 0.20},
            {"n": 3, "max_ratio": 0.18},
            {"n": 4, "max_ratio": 0.16},
        ]
    )

    # ---- deduplication (Stage E) ----
    exact_dedup: bool = True
    fuzzy_dedup: bool = True
    fuzzy_threshold: float = 0.85
    fuzzy_num_perm: int = 128
    paragraph_dedup: bool = True
    paragraph_min_words: int = 12

    # ---- chunking / output (Stage F) ----
    chunking: bool = True
    chunk_max_tokens: int = 1024
    chunk_overlap_tokens: int = 64
    chunk_min_tokens: int = 128
    tokenizer: str = "tiktoken:cl100k_base"
    write_documents: bool = True
    write_chunks: bool = True
    splits: dict = field(
        default_factory=lambda: {"train": 0.98, "val": 0.01, "test": 0.01}
    )
    schema_version: str = "2.0"

    output_format: str = "jsonl"  # jsonl | parquet
    output_dir: str = "data/curated/{topic}"
    shard_size_records: int = 10_000

    # ---- compute ----
    use_gpu: bool = False
    ray_address: str | None = None  # None = local

    # ---- runtime ----
    workdir: str = "."

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Resolve relative paths against the current working directory (project
        # root where the pipeline is run), not the config file's directory.
        base = Path.cwd().resolve()
        workdir = str(base)

        def resolve(value: Any) -> Any:
            if isinstance(value, str):
                p = Path(value)
                if not p.is_absolute():
                    return str((base / p).resolve())
            return value

        # Only known path fields get resolved against the working directory.
        path_fields = {
            "input_urls_file",
            "download_dir",
            "prototype_set_path",
            "output_dir",
            "local_input_dir",
            "jsonl_input_file",
            "perplexity_model_path",
        }
        resolved = {}
        for k, v in raw.items():
            if k in path_fields and isinstance(v, str):
                resolved[k] = resolve(v)
            else:
                resolved[k] = v

        # Interpolate {topic} in output_dir.
        if isinstance(resolved.get("output_dir"), str):
            resolved["output_dir"] = resolved["output_dir"].format(
                topic=resolved.get("topic", "")
            )

        resolved["workdir"] = workdir
        return cls(**resolved)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            k: v for k, v in self.__dict__.items() if k != "workdir"
        }
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    @property
    def resolved_output_dir(self) -> str:
        return self.output_dir


def load_config(path: str | Path) -> PipelineConfig:
    """Module-level wrapper around PipelineConfig.load."""
    return PipelineConfig.load(path)
