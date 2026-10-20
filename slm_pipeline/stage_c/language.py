"""Optional language identification with a confidence score."""

from __future__ import annotations

import math
import re


def detect_language(text: str, *, min_chars: int = 40) -> tuple[str | None, float]:
    """Return ``(language_code, confidence)``.

    ``py3langid`` is preferred (it is small and CPU-only).  Confidence is a
    softmax over the top candidates, not a fake probability from a regex.
    """
    value = re.sub(r"\s+", " ", text or "").strip()
    if len(value) < min_chars:
        return None, 0.0
    try:
        import py3langid  # type: ignore

        ranked = py3langid.rank(value[:4000])
        ranked = [(lang, float(score)) for lang, score in ranked[:5] if math.isfinite(float(score))]
        if not ranked:
            return None, 0.0
        best = ranked[0][1]
        exps = [math.exp(score - best) for _, score in ranked]
        total = sum(exps)
        confidence = exps[0] / total if total else 0.0
        return ranked[0][0], round(confidence, 4)
    except Exception:  # noqa: BLE001, S110 - optional dependency
        pass

    try:
        from lingua import LanguageDetectorBuilder  # type: ignore

        detector = LanguageDetectorBuilder.from_all_languages().build()
        result = detector.detect_language_of(value[:4000])
        if result is None:
            return None, 0.0
        confidences = detector.compute_language_confidence_values(value[:4000])
        confidence = confidences[0].value if confidences else 0.0
        return result.iso_code_639_1.name.lower(), round(float(confidence), 4)
    except Exception:  # noqa: BLE001
        return None, 0.0
