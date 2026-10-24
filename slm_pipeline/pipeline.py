"""Pipeline orchestrator: Stages A -> F.

``run_pipeline`` wires the stages together, persists intermediate JSONL after
each stage (so runs can be resumed), and writes final document/chunk shards plus
a manifest.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import PipelineConfig
from .dataset import DocumentDataset
from .models import Document
from .stage_a import TopicDocumentDownloader, TopicDocumentExtractor, TopicDocumentIterator
from .stage_b import DomainGate, build_default_ml_keywords
from .stage_c.cleaning import (
    build_repeated_line_fingerprints,
    default_cleaning_chain,
    remove_repeated_lines,
)
from .stage_d import QualityFilter
from .stage_e import ExactDuplicates, FuzzyDeduplicator, deduplicate_paragraphs
from .stage_f import (
    Chunk,
    chunk_documents,
    write_chunk_files,
    write_dataset,
    write_manifest,
)
from .stage_f.ordering import rank_documents

STAGE_ORDER = ["extract", "gate", "clean", "quality", "dedup", "output"]


@dataclass
class PipelineResult:
    config: PipelineConfig
    stage_results: dict = field(default_factory=dict)
    final_dataset: DocumentDataset | None = None
    output_files: list[str] = field(default_factory=list)
    chunk_files: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    manifest_path: str | None = None
    elapsed_seconds: float = 0.0


def _load_jsonl(path: str | Path) -> DocumentDataset:
    docs: list[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(Document.from_dict(json.loads(line)))
    return DocumentDataset(docs)


def _save_jsonl(dataset: DocumentDataset, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(path)
    return str(path)


def _read_urls_file(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"URLs file not found: {path}")
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _prune_stale_downloads(download_dir: str | Path, urls: list[str]) -> int:
    """Delete cached downloads that no longer match the current URL list.

    Editing the URL file (dead-link repair, removal) otherwise leaves stale
    files in the download cache: the iterator picks them up, producing
    phantom documents.  A cached file is kept only when its sidecar metadata
    records a URL from the current list (unpaired files are junk).
    """
    allowed = set(urls)
    removed = 0
    directory = Path(download_dir)
    if not directory.exists():
        return 0
    for path in sorted(list(directory.iterdir())):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".html", ".pdf"):
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    meta = {}
            if meta.get("url") not in allowed:
                path.unlink()
                removed += 1
                if meta_path.exists():
                    meta_path.unlink()
        elif path.name.endswith(".meta.json") and suffix == ".json":
            # sidecar of an already-removed asset
            asset = Path(str(path)[: -len(".meta.json")])
            if not asset.exists():
                path.unlink()
    return removed


def _read_sitemap(url: str, *, user_agent: str, timeout: int, max_pages: int | None = None) -> list[str]:
    """Fetch a sitemap (including one level of sitemapindex) and return URLs."""
    import requests
    from lxml import etree

    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    root = etree.fromstring(response.content)
    locs = [el.text.strip() for el in root.xpath("//*[local-name()='loc']") if el.text]
    urls: list[str] = []
    nested = [u for u in locs if u.lower().endswith((".xml", ".xml.gz"))]
    for child in nested[:5]:
        try:
            child_resp = requests.get(child, headers=headers, timeout=timeout)
            child_resp.raise_for_status()
            child_root = etree.fromstring(child_resp.content)
            urls.extend(
                el.text.strip() for el in child_root.xpath("//*[local-name()='loc']")
                if el.text
            )
        except Exception:  # noqa: BLE001, S112
            continue
    urls.extend(u for u in locs if u not in nested)
    # dedupe while preserving order
    seen: set[str] = set()
    out = []
    for value in urls:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out[:max_pages] if max_pages else out


def _extract_from_jsonl(
    records: list[dict],
    extractor: TopicDocumentExtractor,
) -> list:
    results = []
    for item in records:
        if item.get("text"):
            results.append(extractor.extract_text(
                text=item["text"],
                url=item.get("url", ""),
                content_type=item.get("content_type", "text"),
                doc_id=item.get("id"),
                title=item.get("title"),
                fetched_at=item.get("fetched_at"),
            ))
        elif item.get("file_path"):
            results.append(extractor.extract_file(
                file_path=item["file_path"],
                url=item.get("url", ""),
                content_type=item.get("content_type", "html"),
                doc_id=item.get("id"),
                title=item.get("title"),
                fetched_at=item.get("fetched_at"),
            ))
    return results


def _render_js_fallback(
    results: list,
    *,
    config,
    extractor: TopicDocumentExtractor,
    staging_dir: Path,
) -> tuple[list, int]:
    """Re-render thin JavaScript SPA documents with the headless system browser.

    Pages whose server response is an app shell (Kaggle, OSF, OpenML, ...)
    extract to almost no text.  When a system browser is available, those URLs
    are rendered once and the DOM replaces the cached download, so extraction
    recovers the client-rendered content.  Attempt results are memoized in the
    staging directory: any URL already tried is never re-rendered, because a
    re-render would reproduce the same DOM.  ``render_max_pages`` caps how
    many pages one Stage A run may render, so huge URL batches with many
    JS shells cannot turn into hours of browser launches; unprocessed URLs
    simply stay thin and get their chance in the next run.
    """
    from .stage_a.browser import find_chrome, render_page

    chrome = find_chrome(config.chrome_binary)
    if chrome is None:
        print("  browser_fallback enabled but no chrome/chromium on PATH — skipped")
        return results, 0

    attempts_path = staging_dir / "browser_render_attempts.json"
    attempts: dict[str, dict] = {}
    if attempts_path.exists():
        try:
            attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            attempts = {}

    out: list = []
    recovered = 0
    launched = 0
    # Hosts where rendering repeatedly fails (JS hangs until the deadline) are
    # skipped after 3 fruitless tries, so a hopeless SPA host cannot consume
    # the render budget page by page.
    host_book = attempts.get("__host_fails__", {})
    for r in results:
        words = len((r.document.text or "").split())
        if words >= config.render_min_words:
            out.append(r)
            continue
        url = r.document.url
        if url in attempts:
            # Already processed in an earlier run: the rendered DOM (if any)
            # is in the download cache, so re-extraction sees the same text.
            out.append(r)
            continue
        host = urllib.parse.urlparse(url).netloc.lower()
        book = host_book.get(host) or {"tries": 0, "helped": 0}
        if book["tries"] >= 3 and book["helped"] == 0:
            out.append(r)
            continue
        if launched >= config.render_max_pages:
            out.append(r)
            continue
        launched += 1
        html = render_page(url, chrome_path=chrome,
                           timeout_seconds=config.render_timeout_seconds)
        if not html:
            attempts[url] = {"helped": False, "words": words, "error": "render failed"}
            book["tries"] += 1
            host_book[host] = book
            out.append(r)
            continue
        retry = extractor.extract_text(
            text=html, url=url, doc_id=r.document.id,
            fetched_at=r.document.fetched_at,
        )
        retry_words = len((retry.document.text or "").split())
        if retry_words > words:
            # Persist the rendered DOM in the download cache so later runs
            # extract from the file instead of launching the browser again.
            cache = TopicDocumentDownloader(download_dir=config.download_dir)._cache_path(url, False)
            if cache.suffix == ".html":
                cache.write_bytes(html.encode("utf-8"))
            attempts[url] = {"helped": True, "words": retry_words}
            book["tries"] += 1
            book["helped"] += 1
            host_book[host] = book
            recovered += 1
            out.append(retry)
        else:
            attempts[url] = {"helped": False, "words": words, "error": "no content in DOM"}
            book["tries"] += 1
            host_book[host] = book
            out.append(r)

    attempts["__host_fails__"] = host_book
    attempts_path.write_text(json.dumps(attempts, indent=1, sort_keys=True))
    return out, recovered


def _stage_index(name: str) -> int:
    return STAGE_ORDER.index(name)


def run_pipeline(
    config: PipelineConfig | str | Path,
    *,
    stage: Literal["all", "extract", "gate", "clean", "quality", "dedup", "output"] = "all",
    resume_from: str | None = None,
) -> PipelineResult:
    if isinstance(config, (str, Path)):
        from .config import load_config

        config = load_config(config)

    t0 = time.time()
    result = PipelineResult(config=config)

    workdir = Path(config.workdir)
    staging_dir = workdir / ".pipeline_staging" / config.topic
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.resolved_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_from and resume_from not in STAGE_ORDER:
        raise ValueError(f"Unknown resume stage: {resume_from}")
    start_index = _stage_index(resume_from) if resume_from else 0

    def should_run(name: str) -> bool:
        if stage != "all":
            return stage == name
        return _stage_index(name) >= start_index

    def load_previous(name: str) -> DocumentDataset:
        previous = {
            "extract": None,
            "gate": "stage_A_extracted.jsonl",
            "clean": "stage_B_gated.jsonl",
            "quality": "stage_C_cleaned.jsonl",
            "dedup": "stage_D_quality.jsonl",
            "output": "stage_E_deduped.jsonl",
        }[name]
        if previous is None:
            raise RuntimeError("No earlier stage to load for extraction")
        path = staging_dir / previous
        if not path.exists():
            raise FileNotFoundError(f"Missing intermediate for {name}: {path}")
        return _load_jsonl(path)

    dataset: DocumentDataset | None = None

    # ------------------------------------------------------------------ A: extract
    if should_run("extract"):
        print("[Stage A] Ingesting & extracting ...")
        urls: list[str] = []
        download_results = []
        extraction_results = []

        source_type = (config.source_type or "urls").lower()
        extractor = TopicDocumentExtractor(
            source=config.topic,
            extractor=config.extractor,
            min_confidence=config.min_extraction_confidence,
            heading_markup=config.heading_markup,
            math_policy=config.math_policy,
            table_policy=config.table_policy,
            caption_policy=config.caption_policy,
            code_policy=config.code_policy,
        )

        if source_type == "local":
            local_dir = Path(config.local_input_dir or config.download_dir)
            if not local_dir.exists():
                raise FileNotFoundError(f"Local input directory not found: {local_dir}")
            records = list(TopicDocumentIterator(local_dir, source=config.topic).iter_files())
            if config.max_pages:
                records = records[:config.max_pages]
            extraction_results = extractor.extract_records(records)
        elif source_type == "jsonl":
            if not config.jsonl_input_file:
                raise ValueError("source_type=jsonl requires jsonl_input_file")
            raw_records = []
            with open(config.jsonl_input_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_records.append(json.loads(line))
            if config.max_pages:
                raw_records = raw_records[:config.max_pages]
            extraction_results = _extract_from_jsonl(raw_records, extractor)
        else:
            if source_type == "sitemap":
                if not config.sitemap_url:
                    raise ValueError("source_type=sitemap requires sitemap_url")
                urls = _read_sitemap(
                    config.sitemap_url,
                    user_agent=config.user_agent,
                    timeout=config.request_timeout_seconds,
                    max_pages=config.max_pages,
                )
            else:
                urls = _read_urls_file(config.input_urls_file)
                if config.max_pages:
                    urls = urls[:config.max_pages]
                pruned = _prune_stale_downloads(config.download_dir, urls)
                if pruned:
                    print(f"  pruned {pruned} stale cached downloads (no longer in URL list)")

            downloader = TopicDocumentDownloader(
                download_dir=config.download_dir,
                request_delay_seconds=config.request_delay_seconds,
                max_retries=config.max_retries,
                request_timeout_seconds=config.request_timeout_seconds,
                user_agent=config.user_agent,
                respect_robots_txt=config.respect_robots_txt,
            )
            download_results = downloader.download_urls(urls, source=config.topic)
            records = list(TopicDocumentIterator(config.download_dir, source=config.topic).iter_files())
            if config.max_pages:
                records = records[:config.max_pages]
            extraction_results = extractor.extract_records(records)

        # JavaScript-only pages render an app shell with no content text.
        # With a system browser available, recover their DOM-rendered text.
        if config.browser_fallback and source_type == "urls":
            extraction_results, rendered = _render_js_fallback(
                extraction_results,
                config=config,
                extractor=extractor,
                staging_dir=staging_dir,
            )
            if rendered:
                print(f"  browser fallback recovered {rendered} JS-only pages")

        docs: list[Document] = []
        for r in extraction_results:
            doc = r.document
            doc.source = config.topic
            doc.stage = "extracted"
            if not r.ok:
                doc.kept = False
                doc.drop_reason = r.error or "extraction failed"
            docs.append(doc)

        dataset = DocumentDataset(docs)
        _save_jsonl(dataset, staging_dir / "stage_A_extracted.jsonl")
        result.stage_results["extract"] = {
            "source_type": source_type,
            "urls": len(urls) or len(extraction_results),
            "downloaded_ok": sum(1 for dr in download_results if dr.status in {"ok", "skipped"}),
            "downloaded_errors": sum(1 for dr in download_results if dr.status == "error"),
            "extracted_ok": sum(1 for er in extraction_results if er.ok),
            "extracted_fail": sum(1 for er in extraction_results if not er.ok),
            "total_documents": len(dataset),
            "extractors": dict(Counter(
                d.extraction.get("extractor", "none") for d in docs if d.extraction
            )),
        }
        print(f"  extracted {len(dataset)} documents "
              f"({result.stage_results['extract']['extracted_ok']} ok)")

    # ------------------------------------------------------------------ B: domain gate
    if should_run("gate"):
        print("[Stage B] Applying domain gate ...")
        if dataset is None:
            dataset = load_previous("gate")

        keywords = config.keywords or build_default_ml_keywords()
        gate = DomainGate(
            topic=config.topic,
            keywords=keywords,
            keyword_min_hits=config.keyword_min_hits,
            embedding_model=config.embedding_model,
            prototype_set_path=config.prototype_set_path,
            topic_similarity_threshold=config.topic_similarity_threshold,
            enable_embedding=config.enable_embedding,
        )

        gated: list[Document] = []
        keyword_hits_total = 0
        sims: list[float] = []
        for doc in dataset:
            if not doc.kept:
                gated.append(doc)
                continue
            gr = gate.gate(doc)
            doc.kept = gr.keep
            doc.drop_reason = doc.drop_reason or (None if gr.keep else gr.reason)
            doc.stage = "gated"
            doc.metadata["keyword_hits"] = gr.keyword_hits
            doc.metadata["keyword_distinct"] = gr.keyword_distinct
            doc.metadata["gate_layer"] = gr.layer
            if gr.topic_similarity is not None:
                doc.metadata["topic_similarity"] = round(gr.topic_similarity, 4)
                sims.append(gr.topic_similarity)
            keyword_hits_total += gr.keyword_hits
            gated.append(doc)

        dataset = DocumentDataset(gated)
        _save_jsonl(dataset, staging_dir / "stage_B_gated.jsonl")
        kept = sum(1 for d in dataset if d.kept)
        result.stage_results["gate"] = {
            "input": len(dataset),
            "kept": kept,
            "dropped": len(dataset) - kept,
            "avg_topic_similarity": round(sum(sims) / len(sims), 4) if sims else None,
            "avg_keyword_hits": round(keyword_hits_total / max(1, len(dataset)), 2),
        }
        print(f"  gate kept {kept}/{len(dataset)} documents")

    # ------------------------------------------------------------------ C: cleaning
    if should_run("clean"):
        print("[Stage C] Cleaning & normalization ...")
        if dataset is None:
            dataset = load_previous("clean")

        fingerprints: set[str] = set()
        if config.boilerplate_fingerprint:
            fingerprints = build_repeated_line_fingerprints(
                [d for d in dataset if d.kept],
                min_docs=config.boilerplate_min_docs,
            )

        chain = default_cleaning_chain(
            fix_unicode=config.fix_unicode,
            unify_quotes=config.unify_quotes,
            redact_pii=config.redact_pii,
            pii_entities=config.pii_entities,
            pii_person=config.pii_person,
            use_presidio=config.use_presidio,
            language=config.language,
            min_language_confidence=config.min_language_confidence,
            remove_references=config.remove_references,
            remove_captions=config.remove_captions,
            remove_math=config.remove_math,
            math_policy=config.math_policy,
            table_policy=config.table_policy,
            code_policy=config.code_policy,
            caption_policy=config.caption_policy,
            repair_spacing=config.repair_spacing,
            heading_markup=config.heading_markup,
        )

        cleaned: list[Document] = []
        pii_total = 0
        for doc in dataset:
            if not doc.kept:
                cleaned.append(doc)
                continue
            if fingerprints:
                remove_repeated_lines(doc, fingerprints)
            doc = chain(doc)
            doc.stage = "cleaned"
            pii_total += sum(
                int(v) for v in (doc.metadata.get("cleaning_stats", {}).get("pii_counts") or {}).values()
            )
            cleaned.append(doc)

        dataset = DocumentDataset(cleaned)
        _save_jsonl(dataset, staging_dir / "stage_C_cleaned.jsonl")
        kept = sum(1 for d in dataset if d.kept)
        result.stage_results["clean"] = {
            "input": len(dataset),
            "kept": kept,
            "dropped": len(dataset) - kept,
            "pii_redactions": pii_total,
            "boilerplate_fingerprints": len(fingerprints),
        }
        print(f"  cleaned {kept}/{len(dataset)} documents ({pii_total} PII redactions)")

    # ------------------------------------------------------------------ D: quality
    if should_run("quality"):
        print("[Stage D] Quality filtering ...")
        if dataset is None:
            dataset = load_previous("quality")

        qf = QualityFilter(
            min_word_count=config.min_word_count,
            max_word_count=config.max_word_count,
            max_perplexity=config.max_perplexity,
            ngram_filters=config.ngram_filters,
            perplexity_model_path=config.perplexity_model_path,
            min_alpha_ratio=config.min_alpha_ratio,
            max_markup_ratio=config.max_markup_ratio,
            max_reference_ratio=config.max_reference_ratio,
            max_fragment_ratio=config.max_fragment_ratio,
            max_formula_ratio=config.max_formula_ratio,
            max_code_ratio=config.max_code_ratio,
            max_duplicate_line_ratio=config.max_duplicate_line_ratio,
            max_url_ratio=config.max_url_ratio,
            max_nonprintable_ratio=config.max_nonprintable_ratio,
            min_paragraphs=config.min_paragraphs,
            min_sentences=config.min_sentences,
            min_quality_score=config.min_quality_score,
            language=config.language,
            min_language_confidence=config.min_language_confidence,
            require_lead=config.require_lead,
            allow_leading_heading=config.allow_leading_heading,
            min_avg_paragraph_words=config.min_avg_paragraph_words,
            max_heading_ratio=config.max_heading_ratio,
            min_extraction_confidence=config.min_extraction_confidence,
            min_fluency=config.min_fluency,
        )

        quality_filtered: list[Document] = []
        scores: list[float] = []
        for doc in dataset:
            if not doc.kept:
                quality_filtered.append(doc)
                continue
            qr = qf(doc)
            doc.kept = qr.keep
            doc.drop_reason = doc.drop_reason or (None if qr.keep else qr.reason)
            doc.quality_score = qr.quality_score
            doc.stage = "quality"
            doc.metadata["quality_checks"] = qr.checks
            scores.append(qr.quality_score)
            quality_filtered.append(doc)

        dataset = DocumentDataset(quality_filtered)
        _save_jsonl(dataset, staging_dir / "stage_D_quality.jsonl")
        kept = sum(1 for d in dataset if d.kept)
        kept_scores = [d.quality_score or 0.0 for d in dataset if d.kept]
        result.stage_results["quality"] = {
            "input": len(dataset),
            "kept": kept,
            "dropped": len(dataset) - kept,
            "avg_quality_score": round(sum(kept_scores) / max(1, len(kept_scores)), 4),
            "score_min": round(min(kept_scores), 4) if kept_scores else None,
            "score_max": round(max(kept_scores), 4) if kept_scores else None,
            "drop_reasons": dict(Counter(
                d.drop_reason for d in dataset if not d.kept and d.drop_reason
            )),
        }
        print(f"  quality kept {kept}/{len(dataset)} documents")

    # ------------------------------------------------------------------ E: dedup
    if should_run("dedup"):
        print("[Stage E] Deduplication ...")
        if dataset is None:
            dataset = load_previous("dedup")

        exact_dups = 0
        fuzzy_dups = 0
        paragraph_dups = 0
        if config.exact_dedup:
            edr = ExactDuplicates(text_field="text", id_field="id")(dataset.documents)
            exact_dups = edr.exact_duplicates
        if config.fuzzy_dedup:
            fdr = FuzzyDeduplicator(
                threshold=config.fuzzy_threshold,
                num_perm=config.fuzzy_num_perm,
            )(dataset.documents)
            fuzzy_dups = fdr.fuzzy_duplicates
        if config.paragraph_dedup:
            pdr = deduplicate_paragraphs(
                dataset.documents,
                min_words=config.paragraph_min_words,
            )
            paragraph_dups = pdr.paragraph_duplicates

        for d in dataset:
            if d.kept:
                d.stage = "deduped"
        dataset = DocumentDataset(dataset.documents)
        _save_jsonl(dataset, staging_dir / "stage_E_deduped.jsonl")
        kept = sum(1 for d in dataset if d.kept)
        result.stage_results["dedup"] = {
            "input": len(dataset),
            "kept": kept,
            "dropped": len(dataset) - kept,
            "exact_duplicates": exact_dups,
            "fuzzy_duplicates": fuzzy_dups,
            "paragraph_duplicates": paragraph_dups,
        }
        print(f"  dedup kept {kept}/{len(dataset)} documents "
              f"(exact={exact_dups}, fuzzy={fuzzy_dups}, paragraphs={paragraph_dups})")

    # ------------------------------------------------------------------ F: output
    if should_run("output"):
        print("[Stage F] Writing output ...")
        if dataset is None:
            dataset = load_previous("output")

        final = dataset.filter(lambda d: d.kept)

        # Curriculum ordering: rank the whole corpus (older and newly
        # extracted documents together) from base knowledge to advanced and
        # write the output in that order.
        final = DocumentDataset(rank_documents(final.documents))
        result.final_dataset = final

        if config.write_documents:
            result.output_files = write_dataset(
                final,
                output_dir=output_dir,
                output_format=config.output_format,
                shard_size_records=config.shard_size_records,
                prefix="documents",
                schema_version=config.schema_version,
            )
        if config.write_chunks and config.chunking:
            result.chunks = chunk_documents(
                final,
                max_tokens=config.chunk_max_tokens,
                overlap_tokens=config.chunk_overlap_tokens,
                min_tokens=config.chunk_min_tokens,
                tokenizer=config.tokenizer,
                splits=config.splits,
                schema_version=config.schema_version,
            )
            result.chunk_files = write_chunk_files(
                result.chunks,
                output_dir=output_dir / "chunks",
                output_format=config.output_format,
                shard_size_records=config.shard_size_records,
            )

        split_counts = Counter(c.split for c in result.chunks)
        stats = {
            **result.stage_results,
            **final.stats(),
            "documents_written": len(final),
            "chunks_written": len(result.chunks),
            "chunk_tokens": sum(c.token_count for c in result.chunks),
            "splits": dict(split_counts),
            "extractors": dict(Counter(
                d.extraction.get("extractor", "none") for d in final
            )),
            "drop_reasons": dict(Counter(
                d.drop_reason for d in dataset if not d.kept and d.drop_reason
            )),
        }
        config_hash = hashlib.sha256(
            json.dumps(
                {k: v for k, v in config.__dict__.items() if k != "workdir"},
                sort_keys=True, default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        result.manifest_path = write_manifest(
            output_dir,
            result.output_files,
            stats=stats,
            chunk_files=result.chunk_files,
            schema_version=config.schema_version,
            config_hash=config_hash,
        )
        result.stage_results["output"] = {
            "written_records": len(final),
            "written_chunks": len(result.chunks),
            "chunk_tokens": sum(c.token_count for c in result.chunks),
            "files": len(result.output_files),
            "chunk_files": len(result.chunk_files),
            "format": config.output_format,
            "output_dir": str(output_dir),
            "splits": dict(split_counts),
        }
        print(f"  wrote {len(final)} documents and {len(result.chunks)} chunks")
        print(f"  manifest: {result.manifest_path}")

    result.elapsed_seconds = round(time.time() - t0, 2)
    return result
