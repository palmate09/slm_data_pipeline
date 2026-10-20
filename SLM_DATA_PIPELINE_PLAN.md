# SLM Data Collection & Cleaning Pipeline — Implementation Plan

**Goal:** Build an end-to-end pipeline that takes a list of URLs (web pages, docs, arXiv papers) about a single subject (starting with **Machine Learning**) and produces a **tokenization-ready, domain-specific dataset** (clean JSONL/Parquet with a `text` field) for training a Small Language Model (SLM) that knows *only* that subject.

**Target output per topic:** High-quality, deduplicated, domain-filtered text ready for tokenizer → `.bin`/`.npy` → SLM training.

---

## 1. Architecture Overview

```
Input: List of URLs (topic-specific)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE A: Link Ingestion & Extraction                       │
│  • Custom DocumentDownloader / DocumentIterator             │
│  • HTML → text (jusText), PDF → text (pypdf/pdfplumber)     │
│  • Output: raw JSONL records {text, url, title, source}     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE B: Domain Gate (Topic Filter)  ← KEY DIFFERENTIATOR  │
│  • Keyword/rule-based prefilter (ML terms)                  │
│  • Embedding-based classifier vs topic prototype set        │
│  • Keep only ML-relevant documents; discard off-topic       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE C: Cleaning & Normalization (Sequential)             │
│  • UnicodeReformatter (ftfy)                                │
│  • Quotation-mark unification                               │
│  • PII redaction (Presidio: emails, names, phones, etc.)    │
│  • Boilerplate / navigation removal                         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE D: Quality Filtering                                 │
│  • WordCountFilter (min/max words)                          │
│  • RepeatingTopNGramsFilter (spam/repetition removal)       │
│  • PerplexityFilter (KenLM fluency score)                   │
│  • Ensemble quality classifiers (Nemotron-CC style)         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE E: Deduplication                                     │
│  • ExactDuplicates (MD5 hash, GPU via RAPIDS cuDF)          │
│  • Fuzzy dedup (MinHash + LSH, Jaccard similarity)          │
│  • Optional: substring dedup (Google deduplicate-text)      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE F: Output                                            │
│  • Tokenization-ready JSONL (one JSON per line, `text` field)│
│  • Optional Parquet shards                                  │
│  • Metadata preserved: url, source, id, quality_score       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  Tokenizer (Nemotron / HF AutoTokenizer) → SLM Training
```

**Core tech:** NVIDIA NeMo Curator (Apache 2.0), Ray + RAPIDS (GPU acceleration), Python.

---

## 2. Tech Stack

| Layer | Tool / Library | Purpose |
|-------|---------------|---------|
| Curation framework | NVIDIA NeMo Curator | Pipeline stages, DocumentDataset, filters, dedup, modifiers |
| Distributed execution | Ray + NVIDIA RAPIDS (cuDF, cuML, cuGraph) | GPU-accelerated filtering & deduplication |
| HTML extraction | jusText | Boilerplate-resistant article text extraction |
| PDF extraction | pypdf / pdfplumber | arXiv paper text extraction |
| Language ID | FastText | Detect English / filter non-English |
| Unicode fix | ftfy (via UnicodeReformatter) | Fix mojibake / encoding issues |
| Quality scoring | KenLM (perplexity), FineWeb Nemotron/Mixtral Edu classifiers, FastText quality (`fasttext-oh-eli5`) | Fluency & educational quality |
| PII redaction | Microsoft Presidio (via PiiModifier) | Redact emails, names, phones, addresses |
| Deduplication | MD5 hashing (exact), MinHash + LSH (fuzzy) | Remove duplicate / near-duplicate docs |
| Embedding (domain gate) | sentence-transformers (e.g., all-MiniLM-L6-v2) | Topic relevance scoring |
| Output format | JSONL / Parquet | Tokenization-ready |
| Package manager | uv | Recommended by NeMo Curator |

---

## 3. Detailed Pipeline Stages

### Stage A — Link Ingestion & Extraction
- **Input:** `urls.txt` (one URL per line) or a JSON config with URL + optional source label.
- **Custom classes** (modeled on NeMo Curator `DocumentDownloader` / `DocumentIterator` / `DocumentExtractor`):
  - `TopicDocumentDownloader`: downloads HTML/PDF to disk, respects rate limits and `robots.txt`.
  - `TopicDocumentIterator`: reads downloaded files, yields raw records.
  - `TopicDocumentExtractor`: extracts `text`, `title`, `url`, `source`, `id`.
- **HTML:** Fetch → `jusText` extraction → keep only main content (drop nav, ads, footers).
- **PDF (arXiv):** Download PDF → `pypdf`/`pdfplumber` text extraction.
- **Output:** Intermediate JSONL shards: `{text, url, title, source, id, fetched_at}`.

### Stage B — Domain Gate (ML Topic Filter)
*This is the critical stage that ensures the SLM learns only ML and not general knowledge.*

**Two-layer filter:**
1. **Rule-based prefilter (fast, cheap):**
   - Maintain an ML keyword list (e.g., neural network, gradient descent, backpropagation, transformer, PyTorch, scikit-learn, overfitting, loss function, convolution, attention mechanism, etc.).
   - Score document by keyword density / presence.
   - Reject documents below a low threshold (clearly off-topic: cooking, travel, news, etc.).
2. **Embedding-based classifier (accurate):**
   - Encode document with a sentence-embedding model (`all-MiniLM-L6-v2` or smaller for speed).
   - Compare cosine similarity against an ML prototype set (a small curated set of canonical ML texts).
   - Keep documents above `topic_threshold` (configurable, e.g., 0.5–0.7).
3. **Optional quality ensemble:** Apply Nemotron-CC style quality classifiers to reject low-quality ML content.

**Configurable via `topic_config.yaml`:**
```yaml
topic: machine_learning
keywords:
  - "neural network"
  - "gradient descent"
  - "backpropagation"
  - "transformer"
  - "pytorch"
  - "scikit-learn"
embedding_model: sentence-transformers/all-MiniLM-L6-v2
topic_threshold: 0.6
prototype_set_path: configs/ml_prototypes.txt
```

### Stage C — Cleaning & Normalization
Chain via NeMo Curator `Sequential` + `Modify`:
- `UnicodeReformatter` (ftfy): fix encoding/mojibake.
- Custom `QuotationUnifier`: curly → straight quotes.
- Custom boilerplate stripper: remove "Cookie policy", "Subscribe", "Related articles", etc.
- `PiiModifier` (Presidio): redact `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, `IBAN_CODE`, etc. → replace with `[REDACTED]`.

### Stage D — Quality Filtering
Chain via NeMo Curator `ScoreFilter`:
- `WordCountFilter(min_words=50, max_words=100000)`: drop too-short or too-long docs.
- `RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2)`, `(n=3, ...0.18)`, `(n=4, ...0.16)`: drop repetitive/spammy content.
- `PerplexityFilter` with a small KenLM model: drop incoherent/garbage text (lower perplexity = more fluent).
- Optional: Nemotron-CC ensemble classifiers (`FineWebNemotronEduClassifier`, `FineWebMixtralEduClassifier`, `FastTextQualityClassifier`) → keep top quality buckets.

### Stage E — Deduplication
- `ExactDuplicates(id_field="id", text_field="text", hash_method="md5")`: GPU-accelerated exact duplicate removal.
- Fuzzy dedup: MinHash signatures + LSH (Jaccard similarity) to remove near-duplicate tutorials/blog posts that repeat the same ML explanation.
- Optional: Google `deduplicate-text-datasets` (suffix-array substring dedup) for very large corpora.

### Stage F — Output
- `DocumentDataset.to_json(output_path, write_to_filename=True)` → JSONL.
- Each line: `{"text": "...", "url": "...", "source": "...", "id": "...", "quality_score": ...}`.
- This is directly consumable by HuggingFace `datasets` or Nemotron's tokenizer for `.bin`/`.npy` conversion.

---

## 4. Implementation Roadmap (Phases)

| Phase | Milestone | Deliverable |
|-------|-----------|-------------|
| **P1** | MVP ingestion | Custom downloader + extractor: URL → raw JSONL (single machine, no GPU) |
| **P2** | Cleaning pipeline | Add UnicodeReformatter, quotation unifier, PII redaction → clean JSONL |
| **P3** | Quality + dedup | Add WordCountFilter, n-gram filter, PerplexityFilter, ExactDuplicates, fuzzy dedup |
| **P4** | Domain gate | Add ML keyword + embedding topic filter (the differentiator) |
| **P5** | Scale & config | `topic_config.yaml`, CLI entrypoint, Ray multi-GPU execution, Parquet output |
| **P6** | Tokenize + validate | Tokenize output, train tiny SLM checkpoint, evaluate on ML QA vs off-topic |

**Estimated effort:** P1–P4 (core pipeline) ≈ 2–4 weeks depending on team size; P5–P6 (scale + training) ≈ 2–4 weeks.

---

## 5. Configuration (`topic_config.yaml`)

```yaml
# Topic configuration — swap this file to change domain (ML → Physics → Law, etc.)
topic: machine_learning
display_name: Machine Learning

# Ingestion
input_urls_file: data/urls/ml_urls.txt
respect_robots_txt: true
request_delay_seconds: 1.0
max_retries: 3

# Domain gate
embedding_model: sentence-transformers/all-MiniLM-L6-v2
prototype_set_path: configs/ml_prototypes.txt
topic_similarity_threshold: 0.6
keyword_min_hits: 2

# Cleaning
fix_unicode: true
unify_quotes: true
redact_pii: true
pii_entities:
  - PERSON
  - EMAIL_ADDRESS
  - PHONE_NUMBER
  - US_SSN
  - CREDIT_CARD

# Quality filtering
min_word_count: 50
max_word_count: 100000
min_perplexity: null
max_perplexity: 1000.0
ngram_filters:
  - n: 2
    max_ratio: 0.20
  - n: 3
    max_ratio: 0.18
  - n: 4
    max_ratio: 0.16

# Deduplication
exact_dedup: true
fuzzy_dedup: true
fuzzy_threshold: 0.85  # Jaccard

# Output
output_format: jsonl   # jsonl or parquet
output_dir: data/curated/ml
shard_size_records: 10000

# Compute
use_gpu: true
ray_address: null      # null = local; set for multi-node Ray cluster
```

---

## 6. Extensibility to Other Topics

The pipeline is **topic-agnostic** except for Stage B (Domain Gate). To add a new subject:

1. Create a new `topic_config.yaml` (e.g., `physics.yaml`, `law.yaml`, `medicine.yaml`).
2. Provide a topic-specific keyword list.
3. Provide a small prototype text set for embedding similarity (5–20 canonical documents in that domain).
4. Adjust `topic_similarity_threshold` as needed.
5. Re-run Stages A–F; Stages C–F remain unchanged.

Example topic switch:
```
python pipeline.py --config configs/machine_learning.yaml   # → data/curated/ml/
python pipeline.py --config configs/physics.yaml             # → data/curated/physics/
```

---

## 7. SLM Training Readiness

After the pipeline produces curated JSONL:

1. **Tokenize:** Use a tokenizer compatible with your SLM (e.g., Nemotron tokenizer, GPT-NeoX, or HF `AutoTokenizer`) to convert JSONL `text` → token ID arrays (`.bin` / `.npy` / HuggingFace `datasets`).
2. **Train:** Train a small model (e.g., 60M – 1B parameters) from scratch or continue-pretrain a base SLM on the curated domain data.
3. **Evaluate:**
   - **In-domain:** ML quiz / QA accuracy, perplexity on held-out ML text.
   - **Out-of-domain:** Confirm low confidence / high perplexity on non-ML topics (cooking, travel, etc.) to validate specialization.
4. **Ablation:** Train small checkpoints with/without each pipeline stage to measure impact (as NVIDIA did with their 357M model ablation in the NeMo Curator README).

---

## 8. CLI Entry Point (Proposed)

```
python -m slm_pipeline.run \
  --config configs/machine_learning.yaml \
  --stage all          # or: extract, clean, filter, dedup, output
  --resume-from stage  # resume from a specific stage
```

---

## 9. Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Off-topic content leaks into training | Strict domain gate (Stage B) + out-of-domain evaluation |
| Low-quality / spam ML blogs | Quality filters + perplexity + edu-classifier ensemble |
| Duplicate tutorials (same ML concept repeated many times) | Exact + fuzzy + substring deduplication |
| PII / copyrighted content in scraped pages | Presidio PII redaction; review source URLs; add license filter if needed |
| GPU memory OOM on large crawls | Chunked processing, shard output, Ray multi-node scaling |
| Slow embedding inference for millions of docs | Use lightweight embedding model; run keyword prefilter first to discard obvious off-topic docs |

---

## 10. Key References & Citations

- **NeMo Curator** (NVIDIA, GitHub): `https://github.com/NVIDIA-NeMo/Curator` — scalable GPU-accelerated data curation toolkit. Cite as: NVIDIA, "NeMo Curator: GPU-Accelerated Data Curation for Training AI Models," 2024.
- **Nemotron-CC paper** (arXiv:2412.02595, ACL 2025): Su et al., "Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset." Describes classifier ensembling + synthetic data generation pipeline built with NeMo Curator.
- **Nemotron-4 15B Technical Report** (arXiv:2402.16819): Parmar et al. — 15B model trained on 8T tokens curated by NeMo Curator.
- **Nemotron-CC GitHub recipe:** `https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data/curation/nemotron-cc` — 4-step pipeline (download/extract → exact dedup → fuzzy dedup → quality classification → SDG).
- **TinyStories curation tutorial** (NVIDIA Developer Blog, May 2024): End-to-end custom pipeline example (download → clean → filter → dedupe → PII redact).
- **Google deduplicate-text-datasets** (arXiv:2107.06499): Substring deduplication via suffix arrays.
- **jusText:** HTML boilerplate removal. **FastText:** Language identification & quality classification. **Presidio:** PII detection/redaction. **KenLM:** Perplexity scoring.

---

## 11. Immediate Next Steps

1. Save this plan: `SLM_DATA_PIPELINE_PLAN.md` (this file).
2. Set up environment: `uv venv && uv pip install "nemo-curator[text_cpu]"` (CPU MVP) or `text_cuda12` (GPU).
3. Gather an initial `ml_urls.txt` (ML blogs, docs, arXiv links).
4. Implement Phase P1 (URL → raw JSONL) and validate on ~100 links.
5. Iterate through P2–P4, then scale and train.

---

**File location:** `/home/shubham/Desktop/Projects/brainstorm/SLM_DATA_PIPELINE_PLAN.md`  
**Status:** Ready for implementation.  
**Primary domain:** Machine Learning (extensible to any topic via `topic_config.yaml`).
