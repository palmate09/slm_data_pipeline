# SLM Data Collection & Cleaning Pipeline

An end-to-end, topic-specific data curation pipeline for Small Language Model
(SLM) training. It turns URLs, sitemaps, local HTML/PDF files, or JSONL records
into clean, structure-aware documents and token-bounded chunks with deterministic
splits and a manifest.

The pipeline is source-agnostic: the topic is defined by a config file (keywords,
prototype documents, thresholds), while extraction, cleaning, quality checks,
deduplication, and chunking are shared across topics.

## What it does

1. **Extract structure, not just text.** HTML is walked into typed blocks
   (`heading`, `paragraph`, `list`, `code`, `table`, `caption`, `formula`,
   `quote`). Wikipedia uses a dedicated DOM walker so the lead section is kept.
   Generic pages are scored across trafilatura, readability, and jusText.
2. **Clean by block type.** References, citations, captions, navigation,
   maintenance templates, product CTAs, and low-value trailing directories are
   removed. Math is converted to readable text or dropped according to policy;
   code and tables have explicit keep/drop policies.
3. **Redact only direct identifiers by default.** Email, phone, SSN, credit card,
   and IBAN are supported. `PERSON`/address redaction is opt-in through Presidio.
4. **Apply independent quality gates.** Word count, URL/non-printable ratio,
   alphabetic ratio, markup, references, fragments, duplicate lines, formula/code
   ratio, lead presence, language, extraction confidence, and n-gram repetition.
   KenLM perplexity is used only when a real model is configured; the pipeline
   does not fake perplexity with a heuristic.
5. **Deduplicate at three levels.** Normalized exact hashes, MinHash/LSH fuzzy
   clusters with correct representatives, and cross-document paragraph dedup.
6. **Emit token-aware chunks.** Chunks respect a token budget and overlap at
   sentence boundaries, carry heading paths, content hashes, and document-level
   train/val/test assignments.
7. **Write final artifacts.** Document shards, chunk shards by split, and a
   manifest with schema version, config hash, and stage statistics.

## Project layout

```
slm_pipeline/
  stage_a/        # ingestion + structured extraction (HTML/PDF/Markdown/text)
    html_extract.py
    extractor.py
    downloader.py
    iterator.py
  stage_b/        # domain gate: weighted keywords + embedding similarity
  stage_c/        # block-aware cleaning, references, PII, language, artifacts
  stage_d/        # independent quality checks and hard gates
  stage_e/        # exact, fuzzy, and paragraph deduplication
  stage_f/        # token chunking, splits, output, manifest
  pipeline.py     # stage orchestration, source adapters, resume logic
  cli.py          # run/audit/preview/validate commands
  config.py       # topic config loading and path resolution
  dataset.py      # DocumentDataset
  models.py       # Document/Block schema
tests/
configs/
data/urls/
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional extras:

```bash
# Embedding gate (sentence-transformers)
uv pip install -e ".[embedding]"

# Presidio PERSON/ADDRESS redaction
uv pip install -e ".[pii]"

# Parquet output
uv pip install -e ".[parquet]"
```

If you use the project virtualenv without activating it, run commands with
`.venv/bin/python`. Do not mix system `pytest` with the project environment.

## Run

```bash
# Full run
./run_pipeline.sh full

# Resume after a failed/late stage
./run_pipeline.sh resume clean
./run_pipeline.sh resume quality

# Run one stage only (loads the previous intermediate)
.venv/bin/python -m slm_pipeline.cli \
  --config configs/machine_learning.yaml --stage quality
```

Legacy CLI flags are preserved: `--config`, `--stage`, `--resume-from`.

Source adapters are selected in the config:

```yaml
source_type: urls       # urls | sitemap | local | jsonl
input_urls_file: data/urls/ml_urls.txt
# sitemap_url: https://example.com/sitemap.xml
# local_input_dir: data/raw
# jsonl_input_file: data/input.jsonl
```

## Output

By default, final artifacts are written to `output_dir`:

```
data/curated/machine_learning/
  documents_00000.jsonl
  chunks/
    chunks_train_00000.jsonl
    chunks_val_00000.jsonl
    chunks_test_00000.jsonl
  manifest.json
```

Final document schema (one JSON object per line, no intermediate `blocks`):

```json
{
  "id": "...",
  "text": "...",
  "url": "...",
  "source": "machine_learning",
  "title": "...",
  "fetched_at": "...",
  "language": "en",
  "license": "...",
  "quality_score": 0.97,
  "difficulty": 0.14,
  "curriculum_rank": 5,
  "curriculum_level": "foundational",
  "kept": true,
  "stage": "deduped",
  "metadata": {"quality_checks": {...}, "curriculum": {...}},
  "schema_version": "2.0"
}
```

Chunk schema:

```json
{
  "id": "doc-id-c0000",
  "doc_id": "doc-id",
  "text": "...",
  "title": "...",
  "url": "...",
  "source": "machine_learning",
  "language": "en",
  "license": "...",
  "heading_path": ["Approaches", "Reinforcement learning"],
  "chunk_index": 0,
  "token_count": 985,
  "quality_score": 0.97,
  "split": "train",
  "difficulty": 0.14,
  "curriculum_rank": 5,
  "curriculum_level": "foundational",
  "content_hash": "sha256...",
  "schema_version": "2.0"
}
```

Splits are assigned at document level and balanced by size, so every split
lands close to its configured share while no document is ever shared between
splits (no train/eval leakage). Chunks never cross documents.

### Curriculum ordering (base knowledge -> advanced)

Stage F scores the whole corpus - documents extracted in earlier runs and
newly extracted ones together - and writes the output from the foundations to
the advanced material.  The score combines the difficulty of the page topic
(from its title), the technical vocabulary, mathematical notation, code
samples, structure and lexical rarity; it is then rescaled across the corpus
so the levels are meaningful.  Every document and chunk carries:

| field | meaning |
| --- | --- |
| `difficulty` | corpus-relative 0-1 score |
| `curriculum_rank` | 1-based position, easiest first |
| `curriculum_level` | `foundational`, `intermediate`, `advanced`, `expert` |

`metadata.curriculum` keeps the full signal breakdown for inspection.  The
ordering is metadata plus file order: chunk ids and hashes are unchanged, so a
trainer can still shuffle.  `audit` prints the curriculum and `validate`
fails if the chunks of a split are not ordered from base to advanced.

## Inspect and validate

```bash
# Counts, token sizes, splits, artifacts, drop reasons, top documents
.venv/bin/python -m slm_pipeline.cli audit --config configs/machine_learning.yaml

# Preview documents or chunks
.venv/bin/python -m slm_pipeline.cli preview --config configs/machine_learning.yaml \
  --kind chunks --limit 3

# Fail (exit 1) on schema/token/hash/artifact problems, plus unbalanced code
# fences, chunks ending on a heading, and lowercase/mid-sentence starts.
.venv/bin/python -m slm_pipeline.cli validate --config configs/machine_learning.yaml
```

`audit` reports document/chunk structural issue counts. `validate` reports
under-minimum chunks as warnings (a coherent document tail is allowed when
merging it would exceed `chunk_max_tokens`). `audit`, `preview`, and `validate`
accept `--output-dir` if you want to inspect an alternate export.
`validate --strict` treats warnings as errors.

## Configuration highlights

See `configs/machine_learning.yaml` for the full reference.

```yaml
# content policy
content_profile: prose       # prose | technical | code
preserve_code: true
preserve_tables: false
math_policy: text            # text | latex | drop
table_policy: drop           # drop | markdown | csv
code_policy: keep
caption_policy: drop
reference_policy: drop

# PII: direct identifiers only; PERSON is opt-in
pii_entities: [EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, IBAN_CODE]
pii_person: false
use_presidio: false

# quality gates
require_lead: true
allow_leading_heading: true   # tolerate a page title before the prose
min_avg_paragraph_words: 25
max_heading_ratio: 0.5
min_alpha_ratio: 0.6
max_markup_ratio: 0.005
max_duplicate_line_ratio: 0.05
min_quality_score: 0.55

# chunking
chunk_max_tokens: 1024
chunk_overlap_tokens: 64
chunk_min_tokens: 100
tokenizer: tiktoken:cl100k_base
splits: {train: 0.90, val: 0.05, test: 0.05}
```

## Retargeting to a new topic

1. Copy `configs/machine_learning.yaml` to `configs/<topic>.yaml`.
2. Set `topic`, `display_name`, keywords, and a `prototype_set_path` containing
   5–20 canonical documents for the new subject.
3. Tune `topic_similarity_threshold` (default `0.55` in the ML config).
4. Run the pipeline with the new config.

The extraction, cleaning, quality, dedup, and chunking stages do not change.

## Tests

```bash
.venv/bin/python -m pytest -q
# or: make test
```

Tests are offline. The embedding gate test may download the MiniLM model on its
first run; later runs use the local Hugging Face cache.

## Notes

- KenLM is optional. If no `perplexity_model_path` is configured, perplexity is
  reported as unavailable and is not used as a gate. A separate fluency heuristic
  is recorded but only enforced when `min_fluency > 0`.
- `PERSON` redaction requires Presidio and a spaCy model; without them it is
  skipped rather than being silently approximated.
- Intermediate stage snapshots are kept under `.pipeline_staging/<topic>/` for
  resume/debugging. Final document output omits the `blocks` field.
- The downloader follows browser-level redirects (HTML `meta refresh`, JS
  `location.replace`, frameset shells), so documentation sites that redirect
  with a stub page are downloaded as the real article. Cached files that turn
  out to be redirect stubs are re-fetched automatically.
- JavaScript-only pages (SPA shells: Kaggle, OpenML, OSF, ...) render no
  server-side text. With `browser_fallback: true` the pipeline re-renders
  those pages with the headless system browser (chrome/chromium on PATH, no
  new Python dependency), extracts from the DOM, and stores the rendered DOM
  back into the download cache. Render attempts are memoized under
  `.pipeline_staging/<topic>/browser_render_attempts.json` — every URL is
  rendered at most once, and `render_max_pages` caps how many pages one run
  may render so large URL batches cannot stall on browser launches.
- The topic gate compares against the prototype set in
  `configs/ml_prototypes.txt`; widen it when the corpus covers more than the
  core topic so that legitimate sub-field pages are not dropped.

## License

MIT.
