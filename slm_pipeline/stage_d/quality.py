"""Quality filtering with independent checks and explicit hard gates.

The previous filter used a vague "fluency" heuristic as if it were perplexity
and could not distinguish clean article prose from navigation/caption debris.
This version computes measurable checks (markup, references, fragments,
duplication, language, structure) and only uses a perplexity gate when a real
KenLM model is available.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional
    import kenlm

    _HAS_KENLM = True
except ImportError:  # pragma: no cover - optional dependency
    kenlm = None  # type: ignore[assignment]
    _HAS_KENLM = False

from ..stage_c.artifacts import is_reference_line, scan_artifacts

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "for", "with", "as", "by", "at", "from", "into", "about",
    "it", "its", "we", "you", "they", "he", "she", "his", "her", "their",
    "can", "could", "may", "might", "will", "would", "should", "not", "no",
}


@dataclass
class QualityResult:
    keep: bool
    reason: str
    quality_score: float
    word_count: int
    perplexity: float | None = None
    max_ngram_ratio: float | None = None
    checks: dict = field(default_factory=dict)


class WordCountFilter:
    def __init__(self, min_words: int = 50, max_words: int = 100_000) -> None:
        self.min_words = min_words
        self.max_words = max_words

    def __call__(self, document) -> tuple[bool, str, dict]:
        wc = len(document.text.split())
        checks = {"word_count": wc}
        if wc < self.min_words:
            return False, f"too few words ({wc} < {self.min_words})", checks
        if wc > self.max_words:
            return False, f"too many words ({wc} > {self.max_words})", checks
        return True, "word count ok", checks


class RepeatingTopNGramsFilter:
    """Drop documents where a single n-gram dominates (spam/repetition)."""

    def __init__(self, ngram_filters: Iterable[dict]) -> None:
        self.filters = list(ngram_filters)  # each: {"n": int, "max_ratio": float}

    def _top_ngram_ratio(self, text: str, n: int) -> float:
        words = re.findall(r"\S+", text)
        if len(words) < n:
            return 0.0
        ngrams = zip(*[words[i:] for i in range(n)])
        counts = Counter(ngrams)
        if not counts:
            return 0.0
        total = sum(counts.values())
        return max(counts.values()) / total

    def __call__(self, document) -> tuple[bool, str, dict]:
        ratios = {}
        for spec in self.filters:
            n = int(spec["n"])
            max_ratio = float(spec["max_ratio"])
            ratio = self._top_ngram_ratio(document.text, n)
            ratios[f"top{n}gram_ratio"] = round(ratio, 4)
            if ratio > max_ratio:
                return (
                    False,
                    f"repetitive {n}-grams ({ratio:.2f} > {max_ratio})",
                    ratios,
                )
        return True, "ngram repetition ok", ratios


class TextIntegrityFilter:
    """Reject malformed extraction and excessive URLs/non-printable bytes."""

    def __init__(
        self,
        *,
        max_url_ratio: float = 0.08,
        max_nonprintable_ratio: float = 0.01,
    ) -> None:
        self.max_url_ratio = max_url_ratio
        self.max_nonprintable_ratio = max_nonprintable_ratio

    def __call__(self, document) -> tuple[bool, str, dict]:
        text = document.text or ""
        checks = {
            "url_count": len(re.findall(r"https?://\S+|www\.\S+", text, re.IGNORECASE)),
            "nonprintable_ratio": sum(
                not c.isprintable() and c not in "\n\t" for c in text
            ) / max(1, len(text)),
        }
        words = max(1, len(text.split()))
        checks["url_ratio"] = round(checks["url_count"] / words, 4)
        if checks["url_ratio"] > self.max_url_ratio:
            return False, "excessive URLs", checks
        if checks["nonprintable_ratio"] > self.max_nonprintable_ratio:
            return False, "excessive non-printable characters", checks
        return True, "text integrity ok", checks


class PerplexityFilter:
    """Real KenLM perplexity when available; otherwise report unavailable.

    The old code substituted a heuristic "fluency" score and called it
    perplexity.  That is misleading, so the fallback is reported separately and
    is not used as a hard gate unless ``min_fluency`` is explicitly > 0.
    """

    def __init__(
        self,
        *,
        max_perplexity: float | None = 1000.0,
        model_path: str | Path | None = None,
        min_fluency: float = 0.0,
    ) -> None:
        self.max_perplexity = max_perplexity
        self.model_path = Path(model_path) if model_path else None
        self.min_fluency = min_fluency
        self._model = None
        self._use_kenlm = _HAS_KENLM and self.model_path is not None
        if self._use_kenlm:
            try:
                self._model = kenlm.Model(str(self.model_path))  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                self._model = None
                self._use_kenlm = False

    @staticmethod
    def _heuristic_fluency(text: str) -> float:
        words = re.findall(r"\S+", text)
        if not words:
            return 0.0
        alpha_tokens = sum(1 for w in words if re.search(r"[A-Za-z]", w)) / len(words)
        avg_len = sum(len(w) for w in words) / len(words)
        len_score = max(0.0, min(1.0, 1.0 - abs(avg_len - 5.0) / 10.0))
        ttr = len(set(words)) / len(words)
        ttr_score = min(1.0, ttr * 2.0)
        garbage = len(re.findall(r"[^\w\s\.,!?;:'\"\-()\[\]{}]", text)) / max(1, len(text))
        garbage_score = max(0.0, 1.0 - garbage * 5.0)
        return float(round(alpha_tokens * 0.4 + len_score * 0.2 + ttr_score * 0.2
                           + garbage_score * 0.2, 4))

    def __call__(self, document) -> tuple[bool, str, dict]:
        checks: dict = {}
        if self._use_kenlm and self._model is not None:
            try:
                ppl = float(self._model.perplexity(document.text))
                checks["perplexity"] = round(ppl, 2)
                checks["perplexity_method"] = "kenlm"
                if self.max_perplexity is not None and ppl > self.max_perplexity:
                    return False, f"high perplexity ({ppl:.1f} > {self.max_perplexity})", checks
                return True, f"perplexity ok ({ppl:.1f})", checks
            except Exception:  # noqa: BLE001
                checks["perplexity_method"] = "kenlm-error"
        checks["perplexity"] = None
        checks["perplexity_method"] = "unavailable"
        fluency = self._heuristic_fluency(document.text)
        checks["fluency_heuristic"] = fluency
        if self.min_fluency > 0 and fluency < self.min_fluency:
            return False, f"low fluency heuristic ({fluency:.2f} < {self.min_fluency})", checks
        return True, "perplexity gate skipped (no KenLM model)", checks


class QualityFilter:
    """Independent checks + weighted score + hard gates."""

    def __init__(
        self,
        *,
        min_word_count: int = 50,
        max_word_count: int = 100_000,
        max_perplexity: float | None = 1000.0,
        ngram_filters: Iterable[dict] | None = None,
        perplexity_model_path: str | Path | None = None,
        min_alpha_ratio: float = 0.0,
        max_markup_ratio: float = 1.0,
        max_reference_ratio: float = 1.0,
        max_fragment_ratio: float = 1.0,
        max_formula_ratio: float = 1.0,
        max_code_ratio: float = 1.0,
        max_duplicate_line_ratio: float = 1.0,
        max_url_ratio: float = 0.08,
        max_nonprintable_ratio: float = 0.01,
        min_paragraphs: int = 1,
        min_sentences: int = 0,
        min_quality_score: float = 0.0,
        language: str | None = None,
        min_language_confidence: float = 0.9,
        require_lead: bool = False,
        allow_leading_heading: bool = False,
        min_avg_paragraph_words: float = 0.0,
        max_heading_ratio: float = 1.0,
        min_extraction_confidence: float = 0.0,
        min_fluency: float = 0.0,
    ) -> None:
        self.word_filter = WordCountFilter(min_word_count, max_word_count)
        self.ngram_filter = RepeatingTopNGramsFilter(ngram_filters or [])
        self.integrity_filter = TextIntegrityFilter(
            max_url_ratio=max_url_ratio,
            max_nonprintable_ratio=max_nonprintable_ratio,
        )
        self.ppl_filter = PerplexityFilter(
            max_perplexity=max_perplexity,
            model_path=perplexity_model_path,
            min_fluency=min_fluency,
        )
        self.min_alpha_ratio = min_alpha_ratio
        self.max_markup_ratio = max_markup_ratio
        self.max_reference_ratio = max_reference_ratio
        self.max_fragment_ratio = max_fragment_ratio
        self.max_formula_ratio = max_formula_ratio
        self.max_code_ratio = max_code_ratio
        self.max_duplicate_line_ratio = max_duplicate_line_ratio
        self.min_paragraphs = min_paragraphs
        self.min_sentences = min_sentences
        self.min_quality_score = min_quality_score
        self.language = language
        self.min_language_confidence = min_language_confidence
        self.require_lead = require_lead
        self.allow_leading_heading = allow_leading_heading
        self.min_avg_paragraph_words = min_avg_paragraph_words
        self.max_heading_ratio = max_heading_ratio
        self.min_extraction_confidence = min_extraction_confidence

    # ------------------------------------------------------------------ checks
    @staticmethod
    def _prose_blocks(document) -> list:
        blocks = getattr(document, "blocks", None) or []
        return blocks

    def _structural_checks(self, document) -> dict:
        text = document.text or ""
        blocks = self._prose_blocks(document)
        words = text.split()
        word_count = len(words)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        prose_lines: list[str] = []
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if line.startswith(("#", "-", ">", "|")):
                continue
            prose_lines.append(line)
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
        if blocks:
            paragraphs = [b for b in blocks if b.type in {"paragraph", "quote"}]
            paragraph_count = len(paragraphs)
            heading_count = sum(1 for b in blocks if b.type == "heading")
            total_blocks = max(1, len(blocks))
            heading_ratio = heading_count / total_blocks
            avg_paragraph_words = (
                sum(len(b.text.split()) for b in paragraphs) / max(1, len(paragraphs))
            )
            formula_chars = sum(len(b.text) for b in blocks if b.type == "formula")
            code_chars = sum(len(b.text) for b in blocks if b.type == "code")
            char_total = max(1, sum(len(b.text) for b in blocks))
            # Lead: the first meaningful block must be prose, not a heading.
            # With ``allow_leading_heading`` the scan skips a single page
            # title / section heading (docs pages and papers commonly start
            # with "Abstract" or the repository title) and looks at the first
            # prose block that follows it.
            if self.allow_leading_heading:
                prose_blocks = [b for b in blocks if b.type in {"paragraph", "quote"}]
                if prose_blocks:
                    if len(prose_blocks[0].text.split()) >= 8:
                        lead_present = True
                    else:
                        # A short opening line (README tagline, "WIP: ...") is
                        # common on project pages; a real paragraph close
                        # behind it still counts as a lead.
                        lead_present = any(
                            len(b.text.split()) >= 8 for b in prose_blocks[1:3]
                        )
                else:
                    # No prose at all: a structured list (README feature
                    # list) is acceptable content, a stray list item is not.
                    list_block = next((b for b in blocks if b.type == "list"), None)
                    lead_present = bool(
                        list_block
                        and len([i for i in list_block.text.splitlines() if i.strip()]) >= 3
                        and len(list_block.text.split()) >= 15
                    )
            else:
                meaningful = [
                    b for b in blocks
                    if b.type not in {"caption", "navigation", "footnote", "formula"}
                ]
                first = meaningful[0] if meaningful else None
                lead_present = bool(
                    first and first.type in {"paragraph", "quote"}
                    and len(first.text.split()) >= 8
                )
        else:
            paragraph_count = max(1, len(prose_lines))
            heading_count = 0
            heading_ratio = 0.0
            avg_paragraph_words = (
                sum(len(line.split()) for line in prose_lines) / max(1, len(prose_lines))
            )
            formula_chars = len(re.findall(r"\\(?:frac|displaystyle|sum|nabla)", text)) * 20
            code_chars = 0
            char_total = max(1, len(text))
            lead_present = bool(
                prose_lines and len(prose_lines[0].split()) >= 8
                and not prose_lines[0].startswith("#")
            )

        alpha_ratio = sum(1 for w in words if re.search(r"[A-Za-z]", w)) / max(1, word_count)
        stopword_ratio = sum(1 for w in words if re.sub(r"[^\w]", "", w.lower()) in _STOPWORDS) / max(1, word_count)
        markup_chars = len(re.findall(r"[{}\\[\\]<>|]", text))
        markup_ratio = markup_chars / max(1, len(text))
        ref_lines = sum(1 for line in prose_lines if is_reference_line(line))
        reference_ratio = ref_lines / max(1, len(prose_lines))
        fragment_lines = sum(
            1 for line in prose_lines
            if len(line.split()) < 4 and not line.endswith((".", "!", "?", ":", ";"))
        )
        fragment_ratio = fragment_lines / max(1, len(prose_lines))
        duplicate_lines = max(
            0,
            len(prose_lines)
            - len({re.sub(r"\s+", " ", line).casefold() for line in prose_lines}),
        )
        duplicate_line_ratio = duplicate_lines / max(1, len(prose_lines))
        formula_ratio = formula_chars / char_total
        code_ratio = code_chars / char_total
        artifacts = scan_artifacts(text)
        art_count = sum(artifacts.values())
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "heading_count": heading_count,
            "heading_ratio": round(heading_ratio, 4),
            "avg_paragraph_words": round(avg_paragraph_words, 2),
            "lead_present": lead_present,
            "alpha_ratio": round(alpha_ratio, 4),
            "stopword_ratio": round(stopword_ratio, 4),
            "markup_ratio": round(markup_ratio, 6),
            "reference_ratio": round(reference_ratio, 4),
            "fragment_ratio": round(fragment_ratio, 4),
            "duplicate_line_ratio": round(duplicate_line_ratio, 4),
            "formula_ratio": round(formula_ratio, 4),
            "code_ratio": round(code_ratio, 4),
            "artifact_count": art_count,
            "artifacts": artifacts,
        }

    def _language_check(self, document) -> tuple[bool, str, dict]:
        checks: dict = {}
        if not self.language:
            return True, "language check disabled", checks
        lang = getattr(document, "language", None) or document.metadata.get("detected_language")
        conf = float(document.metadata.get("language_confidence", 0.0) or 0.0)
        checks["language"] = lang
        checks["language_confidence"] = conf
        if lang and conf >= self.min_language_confidence and lang != self.language:
            return False, f"language mismatch ({lang} != {self.language})", checks
        return True, "language ok", checks

    def _extraction_check(self, document) -> tuple[bool, str, dict]:
        extraction = getattr(document, "extraction", {}) or {}
        confidence = extraction.get("confidence")
        checks = {"extraction_confidence": confidence, "extractor": extraction.get("extractor")}
        if confidence is not None and confidence < self.min_extraction_confidence:
            return False, f"low extraction confidence ({confidence:.2f} < {self.min_extraction_confidence})", checks
        return True, "extraction confidence ok", checks

    def _score(self, checks: dict) -> float:
        score = 1.0
        if checks["alpha_ratio"] < 0.8:
            score -= (0.8 - checks["alpha_ratio"]) * 0.8
        if checks["stopword_ratio"] < 0.08:
            score -= (0.08 - checks["stopword_ratio"]) * 0.5
        score -= min(0.35, checks["markup_ratio"] * 20.0)
        score -= min(0.35, checks["reference_ratio"] * 4.0)
        score -= min(0.30, checks["fragment_ratio"] * 1.2)
        score -= min(0.30, checks["duplicate_line_ratio"] * 1.5)
        score -= min(0.20, checks["formula_ratio"] * 2.0)
        score -= min(0.15, checks["code_ratio"] * 0.3)
        score -= min(0.20, checks["artifact_count"] / max(1, checks["word_count"]) * 8.0)
        if not checks["lead_present"]:
            score -= 0.10
        if checks["avg_paragraph_words"] < 20:
            score -= min(0.15, (20 - checks["avg_paragraph_words"]) / 20 * 0.15)
        for key, value in checks.items():
            if key.startswith("top") and key.endswith("gram_ratio"):
                score -= max(0.0, value - 0.10) * 0.5
        return float(round(max(0.0, min(1.0, score)), 4))

    # ------------------------------------------------------------------ public
    def __call__(self, document) -> QualityResult:
        checks: dict = {}

        ok, reason, wc_checks = self.word_filter(document)
        checks.update(wc_checks)
        if not ok:
            return QualityResult(False, reason, 0.0, wc_checks["word_count"], checks=checks)

        ok, reason, integrity_checks = self.integrity_filter(document)
        checks.update(integrity_checks)
        if not ok:
            return QualityResult(False, reason, 0.0, wc_checks["word_count"], checks=checks)

        ok, reason, ng_checks = self.ngram_filter(document)
        checks.update(ng_checks)
        if not ok:
            return QualityResult(False, reason, 0.0, wc_checks["word_count"], checks=checks)

        ok, reason, ppl_checks = self.ppl_filter(document)
        checks.update(ppl_checks)
        if not ok:
            return QualityResult(False, reason, 0.0, wc_checks["word_count"],
                                 perplexity=ppl_checks.get("perplexity"), checks=checks)

        structural = self._structural_checks(document)
        checks.update(structural)

        hard_gates = [
            (structural["alpha_ratio"] < self.min_alpha_ratio,
             f"low alphabetic ratio ({structural['alpha_ratio']:.2f} < {self.min_alpha_ratio})"),
            (structural["markup_ratio"] > self.max_markup_ratio,
             f"excessive markup ({structural['markup_ratio']:.4f} > {self.max_markup_ratio})"),
            (structural["reference_ratio"] > self.max_reference_ratio,
             f"excessive references ({structural['reference_ratio']:.3f} > {self.max_reference_ratio})"),
            (structural["fragment_ratio"] > self.max_fragment_ratio,
             f"excessive fragments ({structural['fragment_ratio']:.3f} > {self.max_fragment_ratio})"),
            (structural["formula_ratio"] > self.max_formula_ratio,
             f"excessive formula text ({structural['formula_ratio']:.3f} > {self.max_formula_ratio})"),
            (structural["code_ratio"] > self.max_code_ratio,
             f"excessive code ({structural['code_ratio']:.3f} > {self.max_code_ratio})"),
            (structural["duplicate_line_ratio"] > self.max_duplicate_line_ratio,
             f"excessive duplicate lines ({structural['duplicate_line_ratio']:.3f} > {self.max_duplicate_line_ratio})"),
            (structural["paragraph_count"] < self.min_paragraphs,
             f"too few paragraphs ({structural['paragraph_count']} < {self.min_paragraphs})"),
            (structural["sentence_count"] < self.min_sentences,
             f"too few sentences ({structural['sentence_count']} < {self.min_sentences})"),
            (structural["heading_ratio"] > self.max_heading_ratio,
             f"too heading-heavy ({structural['heading_ratio']:.3f} > {self.max_heading_ratio})"),
            (self.require_lead and not structural["lead_present"],
             "no coherent prose lead"),
            (structural["avg_paragraph_words"] < self.min_avg_paragraph_words,
             f"paragraphs too short ({structural['avg_paragraph_words']:.1f} < {self.min_avg_paragraph_words})"),
        ]
        for failed, message in hard_gates:
            if failed:
                return QualityResult(False, message, 0.0, wc_checks["word_count"],
                                     checks=checks)

        ok, reason, lang_checks = self._language_check(document)
        checks.update(lang_checks)
        if not ok:
            return QualityResult(False, reason, 0.0, wc_checks["word_count"], checks=checks)

        ok, reason, extraction_checks = self._extraction_check(document)
        checks.update(extraction_checks)
        if not ok:
            return QualityResult(False, reason, 0.0, wc_checks["word_count"], checks=checks)

        score = self._score(checks)
        checks["quality_score"] = score
        if score < self.min_quality_score:
            return QualityResult(
                False, f"quality score too low ({score:.3f} < {self.min_quality_score})",
                score, wc_checks["word_count"],
                perplexity=ppl_checks.get("perplexity"), checks=checks,
            )

        document.quality_score = score
        return QualityResult(
            True, "quality ok", score, wc_checks["word_count"],
            perplexity=ppl_checks.get("perplexity"), checks=checks,
        )
