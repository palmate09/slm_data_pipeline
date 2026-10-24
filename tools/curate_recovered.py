"""Clean and quality-filter records produced by recover_high_value.py."""
from __future__ import annotations

import json
from pathlib import Path

from slm_pipeline.models import Block, Document
from slm_pipeline.stage_c.cleaning import default_cleaning_chain
from slm_pipeline.stage_d.quality import QualityFilter


def main() -> int:
    src = Path("/tmp/opencode/recovered_high_value.jsonl")
    out = Path("/tmp/opencode/recovered_curated.jsonl")
    chain = default_cleaning_chain(redact_pii=True, remove_references=True,
                                   remove_captions=True, remove_math=True,
                                   math_policy="text", table_policy="drop",
                                   code_policy="keep", caption_policy="drop")
    quality = QualityFilter(min_word_count=50, max_word_count=100000,
                            max_perplexity=1000, min_alpha_ratio=0.6,
                            max_markup_ratio=0.01, max_reference_ratio=0.01,
                            max_fragment_ratio=0.15, max_formula_ratio=0.02,
                            max_code_ratio=0.7, max_duplicate_line_ratio=0.05,
                            max_url_ratio=0.02, max_nonprintable_ratio=0.02,
                            min_paragraphs=1, min_sentences=5,
                            min_quality_score=0.45, language="en",
                            min_language_confidence=0.9, require_lead=False,
                            allow_leading_heading=True, min_avg_paragraph_words=10,
                            max_heading_ratio=0.7, min_extraction_confidence=0.0,
                            min_fluency=0.0)
    accepted=[]; reasons={}; seen=set()
    for line in src.read_text(errors="replace").splitlines():
        try: r=json.loads(line)
        except Exception: continue
        doc=Document(id=r["id"], text=r.get("text", ""), url=r["url"],
                     title=r.get("title", ""), source="machine_learning_recovery",
                     blocks=[Block(type="paragraph", text=r.get("text", ""))],
                     extraction={"extractor":"direct_recovery", "fetched_url":r.get("fetched_url")})
        doc=chain(doc)
        if not doc.kept:
            reasons[doc.drop_reason or "cleaning"] = reasons.get(doc.drop_reason or "cleaning",0)+1; continue
        qr=quality(doc); doc.quality_score=qr.quality_score
        if not qr.keep:
            reasons[qr.reason or "quality"] = reasons.get(qr.reason or "quality",0)+1; continue
        key=doc.content_hash or doc.text.casefold()
        if key in seen: reasons["duplicate_in_recovery"]=reasons.get("duplicate_in_recovery",0)+1; continue
        seen.add(key); accepted.append(doc.to_dict(include_blocks=False))
    out.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in accepted)+("\n" if accepted else ""))
    print(json.dumps({"input": sum(1 for _ in src.open()), "accepted":len(accepted), "reasons":reasons}, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
