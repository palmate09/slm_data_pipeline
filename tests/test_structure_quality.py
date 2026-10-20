"""Focused tests for the structure-aware extraction/cleaning/chunking path."""

import hashlib
import json
import re

from slm_pipeline.cli import _structure_issues
from slm_pipeline.dataset import DocumentDataset
from slm_pipeline.models import Block, Document
from slm_pipeline.stage_a.html_extract import blocks_to_text, extract_html
from slm_pipeline.stage_b import DomainGate
from slm_pipeline.stage_c.artifacts import is_reference_line, scan_artifacts
from slm_pipeline.stage_c.cleaning import default_cleaning_chain
from slm_pipeline.stage_c.pii import PiiRedactor
from slm_pipeline.stage_d import QualityFilter
from slm_pipeline.stage_e import ExactDuplicates, FuzzyDeduplicator, deduplicate_paragraphs
from slm_pipeline.stage_f import (
    TokenCounter,
    assign_splits,
    chunk_document,
    write_chunks,
    write_dataset,
)

WIKI_HTML = """<html><head><title>Test article - Wikipedia</title></head><body>
<div id="mw-content-text"><div class="mw-parser-output">
<p>This is a lead paragraph about gradient descent and neural network training.
It has enough words to be treated as coherent prose rather than a caption.</p>
<h2>History</h2>
<p>Gradient descent was developed as an optimization method for differentiable
functions and later became central to machine learning.</p>
<pre>for step in range(10):\n    weights -= learning_rate * gradient</pre>
<table class="wikitable"><tr><th>Method</th><th>Use</th></tr>
<tr><td>SGD</td><td>online learning</td></tr></table>
<figure><figcaption>Figure 1: a diagram showing gradient descent steps.</figcaption></figure>
<ul><li>Supervised learning</li><li>Unsupervised learning</li></ul>
<h2>References</h2><ul><li>Reference one</li></ul>
</div></div>
<footer>Cookie policy | Subscribe to our newsletter</footer>
</body></html>"""


def test_wikipedia_structured_extraction_preserves_lead_and_blocks():
    result = extract_html(
        WIKI_HTML,
        url="https://en.wikipedia.org/wiki/Test_article",
        requested_extractor="wikipedia",
        min_confidence=0.0,
    )
    assert result.error is None
    assert result.blocks
    assert result.blocks[0].type == "paragraph"
    assert "lead paragraph" in result.blocks[0].text
    types = {block.type for block in result.blocks}
    assert {"heading", "paragraph", "code", "table", "caption", "list"} <= types
    rendered = blocks_to_text(result.blocks)
    assert "Cookie policy" not in rendered
    assert "Reference one" not in rendered


def test_cleaning_removes_references_captions_math_marketing_and_tail_directory():
    blocks = [
        Block(type="paragraph", text=(
            "Gradient descent is an optimization method.[12] "
            "Its update is x = x - alpha * gradient "
            "{\\displaystyle x=x-\\alpha \\nabla f(x)}."
        )),
        Block(type="heading", text="History", level=2),
        Block(type="paragraph", text="The method has a long history in numerical optimization."),
        Block(type="caption", text="Figure 1: a diagram showing the update rule."),
        Block(type="formula", text="\\frac{\\partial L}{\\partial w}"),
        Block(type="code", text="weights -= learning_rate * gradient"),
        Block(type="heading", text="References", level=2),
        Block(type="list", text="- Smith, J. (1999). Journal of Optimization. pp. 1-20."),
        Block(type="heading", text="Aftermath", level=2),
        Block(type="paragraph", text="Aftermath prose should remain in the cleaned document."),
        Block(type="paragraph", text=(
            "Easily design scalable AI assistants and automate tasks with IBM watsonx Orchestrate."
        )),
        Block(type="paragraph", text=(
            '1. "Some Studies in Machine Learning" IBM Journal, 1959. '
            '2. "Learning Representations" Proceedings of NeurIPS, 2014.'
        )),
        Block(type="heading", text="Software", level=2),
        Block(type="list", text="- Tool A\n- Tool B\n- Tool C"),
    ]
    doc = Document(id="clean-1", text="", url="https://example.com", blocks=blocks)
    cleaned = default_cleaning_chain(
        redact_pii=False,
        remove_references=True,
        remove_captions=True,
        remove_math=True,
        math_policy="text",
        table_policy="drop",
        code_policy="keep",
        caption_policy="drop",
    )(doc)
    text = cleaned.text
    lowered = text.lower()
    assert "references" not in lowered
    assert "smith, j." not in lowered
    assert "figure 1" not in lowered
    assert "ibm watsonx" not in lowered
    assert "some studies in machine learning" not in lowered
    assert "software" not in lowered
    assert "tool a" not in lowered
    assert "{\\displaystyle" not in text
    assert "weights -= learning_rate" in text
    assert "aftermath prose" in lowered
    assert cleaned.metadata["cleaning_stats"]["output_blocks"] > 0


def test_cleaning_repairs_dangling_formula_prose_and_markers():
    blocks = [
        Block(type="paragraph", text=(
            "Gradient descent can be used to solve a system of linear equations"
        )),
        Block(type="paragraph", text=(
            "reformulated as a quadratic minimization problem. If the system "
            "matrix is symmetric, an objective function is defined as the "
            "quadratic function, with minimization of"
        )),
        Block(type="paragraph", text="so that"),
        Block(type="paragraph", text="For a general real matrix, linear least squares define"),
        Block(type="paragraph", text=(
            "For example, for a symmetric matrix, a simple algorithm can be as follows,"
        )),
        Block(type="paragraph", text=(
            "To avoid multiplying twice per iteration, we note that the update "
            "implies a residual update, which gives the traditional algorithm,"
        )),
        Block(type="paragraph", text=(
            "This can be done with any of a variety of line search algorithms."
        )),
        Block(type="paragraph", text="The softmax activation function is."),
        Block(type="paragraph", text=(
            'A special token is used as a generic stand-in, written as "" for "unknown".'
        )),
        Block(type="paragraph", text="Optimization methods are useful.[vague]."),
        Block(type="paragraph", text=(
            "The rule { o n i o n s, p o t a t o e s } implies { b u r g e r }."
        )),
        Block(type="paragraph", text=(
            "The output uses s o f t m a x and L a y e r N o r m."
        )),
        Block(type="paragraph", text=(
            "[Note, if any neurons were not connected, they would be independent.]"
        )),
        Block(type="paragraph", text=(
            "Convolutional networks were superseded for speech by LSTM. but are "
            "more successful in vision."
        )),
        Block(type="paragraph", text=(
            "Its embedding vector is Embed (3) = [0, 1] M The token vectors are "
            "added to their positional encodings."
        )),
        Block(type="paragraph", text=(
            "The un-embedding layer is UnEmbed (x) = softmax (x W + b) The matrix "
            "has shape (d, |V|)."
        )),
        Block(type="paragraph", text=(
            "The output is x + Sublayer (LayerNorm (x)) The original model used "
            "a different convention."
        )),
        Block(type="paragraph", text=(
            "Here,, and are the model parameters: adjusting them changes how "
            "heavily each variable is weighted."
        )),
        Block(type="paragraph", text=(
            "IBM itself maintains and updates a significant library of tutorials "
            "for beginners and advanced practitioners alike."
        )),
        Block(type="paragraph", text=(
            "This is why backpropagation needs a differentiable activation function. "
            "(Nevertheless, ReLU is non-differentiable at zero and remains popular, e.g."
        )),
        Block(type="paragraph", text=(
            "Substituting Eq. 2, Eq. 3 and Eq. 4 in Eq. 1 we obtain: if the "
            "activation is logistic and the error is squared."
        )),
        Block(type="paragraph", text=(
            "The philosopher distinguishes five types of machinic capture: "
            "(1) gamification, (2) trapping and tracking (e.g. CAPTCHAs for "
            "image recognition or click-tracking on search results pages), "
            "(3) social motivations and (5) clickwork."
        )),
        Block(type="paragraph", text=(
            "Robot learning is inspired by many machine learning methods (e.g."
        )),
        Block(type="paragraph", text=(
            "The formula is x = y z The next sentence continues the explanation."
        )),
        Block(type="paragraph", text=(
            "The matrix is B = (0 1 2) in other words, it is a bias matrix."
        )),
        Block(type="paragraph", text=(
            "Stay up to date on industry news with the Think Newsletter, "
            "delivered twice weekly."
        )),
    ]
    doc = Document(id="dangling", text="", url="u", blocks=blocks)
    cleaned = default_cleaning_chain(
        redact_pii=False,
        remove_references=True,
        remove_captions=True,
        remove_math=True,
        math_policy="text",
        table_policy="drop",
        code_policy="keep",
        caption_policy="drop",
    )(doc)
    text = cleaned.text
    assert "Gradient descent can be used" in text
    assert "reformulated as a quadratic minimization problem." in text
    assert "minimization of so that" not in text
    assert "linear least squares define" not in text
    assert "as follows" not in text
    assert "traditional algorithm." in text
    assert "The softmax activation function is." not in text
    assert 'written as ""' not in text
    assert "vague" not in text.lower()
    assert not re.search(r"(?<!\.)\.\.(?!\.)", text)
    assert "onions, potatoes" in text
    assert "burger" in text
    assert "softmax and LayerNorm" in text
    assert "[Note," not in text
    assert "Note, if any neurons" in text
    assert "LSTM, but are more successful" in text
    assert "Embed (3) = [0, 1]. The token vectors" in text
    assert "UnEmbed (x) = softmax (x W + b). The matrix" in text
    assert "LayerNorm (x)). The original model" in text
    assert "M The" not in text
    assert "Here,," not in text
    assert "and are the model parameters" not in text
    assert "IBM itself maintains" not in text
    assert "Nevertheless" not in text
    assert "we obtain:" not in text
    assert "CAPTCHAs for image recognition" in text
    assert "clickwork" in text
    assert "Robot learning is inspired" not in text
    assert "x = y z. The next sentence" in text
    assert "B = (0 1 2). In other words" in text
    assert "Think Newsletter" not in text
    assert not any(line[:1].islower() for line in text.splitlines() if line)


def test_cleaning_drops_trailing_short_example_directory():
    prose = (
        "Gradient descent is a method for unconstrained mathematical optimization. "
        "It uses the gradient of the objective function to choose a search direction."
    )
    blocks = [
        Block(type="paragraph", text=prose),
        Block(type="heading", text="Examples", level=2),
        Block(type="list", text="- Yang-Mills flow\n- Yang-Mills-Higgs flow\n- Seiberg-Witten flow"),
    ]
    doc = Document(id="examples", text="", url="u", blocks=blocks)
    cleaned = default_cleaning_chain(
        redact_pii=False,
        remove_references=True,
        remove_captions=True,
        remove_math=True,
        math_policy="text",
        table_policy="drop",
        code_policy="keep",
        caption_policy="drop",
    )(doc)
    assert "Gradient descent is a method" in cleaned.text
    assert "Examples" not in cleaned.text
    assert "Yang-Mills flow" not in cleaned.text
    assert not cleaned.text.rstrip().endswith("-")


def test_pii_redacts_direct_identifiers_without_destroying_bibliography():
    redactor = PiiRedactor()
    text = (
        "ISBN 978-0-12-374856-0; doi:10.1007/978-94-009-0279-4_9; "
        "arXiv:1412.1193; pp. 151-170. "
        "Email alice@example.com, phone +1 555-123-4567, "
        "SSN 123-45-6789, card 4111 1111 1111 1111."
    )
    cleaned, counts = redactor.redact(text)
    assert "978-0-12-374856-0" in cleaned
    assert "10.1007/978-94-009-0279-4_9" in cleaned
    assert "1412.1193" in cleaned
    assert "151-170" in cleaned
    assert "alice@example.com" not in cleaned
    assert "555-123-4567" not in cleaned
    assert "123-45-6789" not in cleaned
    assert "4111 1111 1111 1111" not in cleaned
    assert counts["EMAIL_ADDRESS"] == 1
    assert counts["PHONE_NUMBER"] == 1
    assert counts["US_SSN"] == 1
    assert counts["CREDIT_CARD"] == 1


def test_reference_line_detection_catches_numbered_bibliography():
    assert is_reference_line(
        '1. "Some Studies in Machine Learning" IBM Journal, 1959.'
    )
    assert not is_reference_line(
        "1. Train the model on a small dataset, then evaluate it on a validation set."
    )


def _prose_text(words: int = 80) -> str:
    base = (
        "Machine learning algorithms learn patterns from data and generalize to new examples. "
        "Gradient descent updates model parameters in the direction of the negative gradient. "
    )
    tokens = base.split()
    return " ".join((tokens * ((words // len(tokens)) + 1))[:words])


def test_quality_requires_a_prose_lead_and_ignores_code_duplicates():
    lead = _prose_text(80)
    heading_first = Document(
        id="h1", text="## Introduction\n\n" + lead, url="u",
        blocks=[Block(type="heading", text="Introduction", level=2), Block(type="paragraph", text=lead)],
    )
    qf = QualityFilter(
        min_word_count=20, max_word_count=100000, ngram_filters=[],
        max_perplexity=None, require_lead=True, min_paragraphs=1,
        min_sentences=0, min_quality_score=0.0, max_markup_ratio=1.0,
    )
    result = qf(heading_first)
    assert result.keep is False
    assert "lead" in result.reason

    code = "for x in data:\n    print(x)\nfor x in data:\n    print(x)\nfor x in data:\n    print(x)"
    doc = Document(
        id="c1", url="u",
        blocks=[Block(type="paragraph", text=lead), Block(type="code", text=code)],
        text=blocks_to_text([Block(type="paragraph", text=lead), Block(type="code", text=code)]),
    )
    qf_code = QualityFilter(
        min_word_count=20, max_word_count=100000, ngram_filters=[],
        max_perplexity=None, require_lead=True, min_paragraphs=1,
        min_sentences=0, min_quality_score=0.0, max_markup_ratio=1.0,
        max_duplicate_line_ratio=0.01, max_code_ratio=1.0,
    )
    result_code = qf_code(doc)
    assert result_code.keep is True, result_code.reason
    assert result_code.checks["duplicate_line_ratio"] == 0.0


def test_quality_drops_heading_heavy_tutorial_index():
    lead = _prose_text(80)
    blocks = [Block(type="paragraph", text=lead)]
    for index in range(10):
        blocks.append(Block(type="heading", text=f"Module {index}", level=2))
        blocks.append(Block(type="paragraph", text="This section introduces the topic."))
    doc = Document(
        id="index", url="u", blocks=blocks, text=blocks_to_text(blocks),
    )
    qf = QualityFilter(
        min_word_count=20, max_word_count=100000, ngram_filters=[],
        max_perplexity=None, require_lead=True, min_paragraphs=2,
        min_sentences=0, min_quality_score=0.0, max_markup_ratio=1.0,
        min_avg_paragraph_words=25.0, max_heading_ratio=0.5,
    )
    result = qf(doc)
    assert result.keep is False
    assert "short" in result.reason or "heading-heavy" in result.reason


def test_gate_counts_occurrences_not_only_distinct_keywords():
    gate = DomainGate(
        topic="ml", keywords=["gradient descent", "neural network"],
        keyword_min_hits=2, enable_embedding=False,
    )
    doc = Document(
        id="g1", url="u", text="Gradient descent is useful. Gradient descent is common.",
        title="Gradient descent",
    )
    result = gate.gate(doc)
    assert result.keep is True
    assert result.keyword_hits >= 2
    assert result.keyword_distinct == 1


def test_exact_and_fuzzy_dedup_never_report_dropped_cluster_members_as_kept():
    docs = [
        Document(id="a", text="same machine learning text", url="u"),
        Document(id="b", text="same machine learning text", url="u"),
        Document(id="c", text="unique neural network content", url="u"),
    ]
    exact = ExactDuplicates()(docs)
    assert set(exact.kept_ids) == {"a", "c"}
    assert set(exact.dropped_ids) == {"b"}
    assert all(doc.kept == (doc.id in exact.kept_ids) for doc in docs)

    docs2 = [
        Document(id="a", text=("gradient descent optimizes neural network weights "
                               "through backpropagation over many layers"), url="u"),
        Document(id="b", text=("gradient descent optimizes neural network weights "
                               "through backpropagation over many layers and works"), url="u"),
        Document(id="c", text="baking cookies in a hot oven", url="u"),
    ]
    fuzzy = FuzzyDeduplicator(threshold=0.5, num_perm=64)(docs2)
    assert len(fuzzy.kept_ids) == 2
    assert fuzzy.dropped_ids
    assert all(doc.kept == (doc.id in fuzzy.kept_ids) for doc in docs2)


def test_paragraph_dedup_keeps_best_document_copy():
    paragraph = (
        "Machine learning is a field of study in artificial intelligence concerned "
        "with statistical algorithms that learn from data and generalize to unseen data."
    )
    unique_high = (
        "This high-quality section explains validation, regularization and practical "
        "model selection in enough detail to stand on its own as prose. It also "
        "describes how to compare candidate models with a held-out validation set "
        "and why a clean test split is essential for honest evaluation."
    )
    unique_low = (
        "This lower-quality section still contains enough independent prose to "
        "survive removal of the shared paragraph without leaving an empty document. "
        "It mentions cross-validation, feature scaling and error analysis so the "
        "section remains useful even after exact duplicate text is removed."
    )
    high = Document(
        id="high", url="u", text=paragraph + "\n\n" + unique_high,
        quality_score=0.9,
        blocks=[Block(type="paragraph", text=paragraph), Block(type="paragraph", text=unique_high)],
    )
    low = Document(
        id="low", url="u", text=paragraph + "\n\n" + unique_low,
        quality_score=0.5,
        blocks=[Block(type="paragraph", text=paragraph), Block(type="paragraph", text=unique_low)],
    )
    result = deduplicate_paragraphs([low, high], min_words=8)
    assert result.paragraph_duplicates == 1
    assert len(high.blocks) == 2
    assert len(low.blocks) == 1
    assert "Machine learning is a field" not in low.text


def test_chunk_schema_bounds_hashes_and_deterministic_splits(tmp_path):
    docs = []
    for i in range(6):
        body = " ".join([
            f"Document {i} explains machine learning and neural networks.",
            "Gradient descent minimizes a loss function over many iterations.",
            "Transformers use attention to model relationships in sequences.",
        ] * 20)
        docs.append(Document(
            id=f"doc-{i}", url=f"https://example.com/{i}", title=f"Doc {i}",
            text=body,
            blocks=[Block(type="paragraph", text=body)],
        ))
    splits = assign_splits(docs, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert set(splits.values()) == {"train", "val", "test"}
    assert assign_splits(list(reversed(docs)), {"train": 0.8, "val": 0.1, "test": 0.1}) == splits

    chunk = chunk_document(
        docs[0], max_tokens=120, overlap_tokens=20, min_tokens=10,
        tokenizer="words", split=splits["doc-0"],
    )[0]
    assert chunk.token_count <= 120
    assert chunk.token_count == TokenCounter("words").count(chunk.text)
    assert chunk.split == splits["doc-0"]
    assert chunk.content_hash == hashlib.sha256(chunk.text.encode()).hexdigest()
    assert chunk.id.startswith("doc-0-c")
    assert isinstance(chunk.heading_path, list)

    chunks = [chunk]
    files = write_chunks(chunks, output_dir=tmp_path, output_format="jsonl")
    assert files
    with open(files[0], encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    assert record["id"] == chunk.id
    assert record["token_count"] <= 120
    assert "text" in record and "content_hash" in record

    # A shorter/different later run must not leave stale shards behind.
    stale_chunk = tmp_path / "chunks_legacy_00000.jsonl"
    stale_chunk.write_text("old", encoding="utf-8")
    write_chunks(chunks, output_dir=tmp_path, output_format="jsonl")
    assert not stale_chunk.exists()

    stale_doc = tmp_path / "shard_00000.jsonl"
    stale_doc.write_text("old", encoding="utf-8")
    doc_files = write_dataset(
        DocumentDataset(docs[:1]), output_dir=tmp_path, output_format="jsonl",
    )
    assert doc_files
    assert not stale_doc.exists()


def _chunk_lines(text: str) -> list[str]:
    return [line for line in text.strip().splitlines() if line.strip()]


def test_chunking_preserves_content_headings_and_code_fences():
    long_paragraph = " ".join([
        "Gradient descent is an optimization method that updates parameters.",
        "Each step uses the negative gradient of the loss function.",
        "A small learning rate makes the optimization stable.",
    ] * 18)
    code = (
        "def train(model, data):\n"
        "    for batch in data:\n"
        "        loss = model(batch)\n"
        "        loss.backward()\n"
        "    return model"
    )
    blocks = [
        Block(type="heading", text="Introduction", level=2),
        Block(type="paragraph", text=long_paragraph),
        Block(type="heading", text="Implementation", level=2),
        Block(type="code", text=code),
        Block(type="heading", text="Details", level=3),
        Block(type="paragraph", text=(
            "The unique closing paragraph explains validation and regularization "
            "with enough context to remain useful on its own."
        )),
        Block(type="list", text="- alpha\n- beta\n- gamma"),
        Block(type="heading", text="Summary", level=2),
        Block(type="paragraph", text=(
            "A final prose paragraph ensures the last heading is carried with content."
        )),
    ]
    doc = Document(
        id="chunk-structure", url="u", title="Chunk structure", blocks=blocks,
        text=blocks_to_text(blocks),
    )
    chunks = chunk_document(
        doc, max_tokens=90, overlap_tokens=18, min_tokens=10, tokenizer="words",
    )
    assert chunks
    combined = "\n".join(chunk.text for chunk in chunks)
    for marker in (
        "Introduction", "Implementation", "Details", "Summary",
        "def train", "return model", "unique closing paragraph", "final prose paragraph",
    ):
        assert marker in combined
    for chunk in chunks:
        lines = _chunk_lines(chunk.text)
        assert chunk.token_count == TokenCounter("words").count(chunk.text)
        assert chunk.token_count <= 90
        assert sum(line.lstrip().startswith("```") for line in lines) % 2 == 0
        assert not lines[-1].lstrip().startswith("#")
        first = lines[0].lstrip()
        if not first.startswith(("```", "#", "- ", "* ")):
            assert not first[:1].islower(), first
            assert first[:1] not in ",;:.)]}", first


def test_chunking_overlap_does_not_splice_sentences_or_exceed_budget():
    sentences = [
        "Gradient descent minimizes a differentiable loss function.",
        "Backpropagation computes gradients for each network layer.",
        "Transformers use attention to model long-range dependencies.",
        "Regularization reduces overfitting on small training sets.",
    ] * 22
    doc = Document(
        id="chunk-overlap", url="u", title="Overlap",
        text=" ".join(sentences),
        blocks=[Block(type="paragraph", text=" ".join(sentences))],
    )
    chunks = chunk_document(
        doc, max_tokens=75, overlap_tokens=25, min_tokens=5, tokenizer="words",
    )
    assert len(chunks) > 1
    for chunk in chunks:
        lines = _chunk_lines(chunk.text)
        assert chunk.token_count <= 75
        assert not lines[-1].lstrip().startswith("#")
        assert sum(line.lstrip().startswith("```") for line in lines) % 2 == 0
        assert lines[0][:1].isupper(), lines[0]
        assert lines[0][:1] not in ",;:.)]}"
    # Overlap is context only: it must not introduce a heading or code fence.
    for chunk in chunks[1:]:
        assert not chunk.text.lstrip().startswith(("#", "```"))


def test_structure_validation_flags_fences_headings_and_lowercase_starts():
    assert _structure_issues("## Heading\n\nA complete sentence.") == []
    assert "unbalanced code fence" in _structure_issues("```python\nprint(1)")
    assert "ends with a dangling heading" in _structure_issues("A sentence.\n\n## Heading")
    assert "starts with lowercase text" in _structure_issues("mid-sentence fragment.")
    assert any(
        "closing punctuation" in issue for issue in _structure_issues(", a spliced sentence.")
    )


def test_artifact_scan_ignores_code_but_flags_prose():
    code = (
        "Run the demo:\n\n```bash\ngit clone https://github.com/karpathy/nanoGPT.git\n"
        "python train.py --append-eod <eod>\n```\n"
    )
    assert scan_artifacts(code) == {}

    prose = "See https://example.com/model for the weights."
    counts = scan_artifacts(prose)
    assert counts.get("url") == 1

    # A language switcher row and a stray "Română" are not mojibake.
    assert not scan_artifacts("English | Deutsch | Română").get("mojibake")
    assert scan_artifacts("itâ€™s broken").get("mojibake")


def test_cleaning_strips_markup_redaction_markers_and_sponsor_ctas():
    blocks = [
        Block(type="paragraph", text=(
            "The <strong>transformer</strong> uses {{attention}} weights and "
            "[[Wikilink|attention]] blocks. See the [paper](https://arxiv.org/abs/1706.03762) "
            "for the derivation."
        )),
        Block(type="paragraph", text=(
            "Checkpoint requests are handled by the maintainers of the repository."
        )),
        Block(type="paragraph", text=(
            "Get started with $30/mo in free credits today at https://modal.com. "
            "We have been getting an amazing support from Modal's team."
        )),
        Block(type="paragraph", text=(
            "All nanoGPT experiments are powered by GPUs on Lambda labs, my favorite "
            "Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!"
        )),
        Block(type="list", text=(
            "- Vocab: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json\n"
            "- Merge: https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt\n"
            "- Distributed training with ZeRO and 3D parallelism"
        )),
        Block(type="paragraph", text=(
            "English | 简体中文 | 繁體中文 | 한국어 | Español | Deutsch | Italiano"
        )),
        Block(type="code", text="git clone https://github.com/karpathy/nanoGPT.git"),
    ]
    doc = Document(id="readme-1", text="", url="https://github.com/karpathy/nanoGPT", blocks=blocks)
    cleaned = default_cleaning_chain(
        redact_pii=False,
        remove_references=True,
        remove_captions=True,
        remove_math=True,
        math_policy="text",
        table_policy="drop",
        code_policy="keep",
        caption_policy="drop",
    )(doc)
    text = cleaned.text
    assert "<strong>" not in text
    assert "{{attention}}" not in text
    assert "[[Wikilink|attention]]" not in text
    assert "https://arxiv.org/abs/1706.03762" not in text
    assert "https://s3.amazonaws.com" not in text
    assert "modal.com" not in text
    assert "sponsoring nanoGPT" not in text
    assert "简体中文" not in text
    assert "Distributed training with ZeRO" in text
    # Code is preserved verbatim, including its URL.
    assert "git clone https://github.com/karpathy/nanoGPT.git" in text
    assert scan_artifacts(text) == {}


def test_cleaning_drops_sentences_with_redaction_markers():
    blocks = [
        Block(type="paragraph", text=(
            "The model was trained on a large corpus of scientific papers. "
            "Interested people can email us at [REDACTED] to arrange access. "
            "The checkpoint is published on the project page."
        )),
    ]
    doc = Document(id="pii-1", text="", url="https://example.com", blocks=blocks)
    cleaned = default_cleaning_chain(
        redact_pii=True,
        remove_references=True,
        remove_captions=True,
        remove_math=True,
        math_policy="text",
        table_policy="drop",
        code_policy="keep",
        caption_policy="drop",
    )(doc)
    assert "[REDACTED]" not in cleaned.text
    assert "The model was trained on a large corpus" in cleaned.text
    assert "The checkpoint is published on the project page." in cleaned.text


def test_pii_keeps_model_parameter_counts():
    redactor = PiiRedactor()
    text, counts = redactor.redact(
        "GPT-2 used 48 layers and d_model 1600 (vs. original 12 layers and "
        "d_model 768). ~1.542B params and 117M parameters"
    )
    assert "1.542B" in text
    assert "117M" in text
    assert not counts.get("PHONE_NUMBER")


def test_pii_redacts_email_at_end_of_sentence():
    redactor = PiiRedactor()
    text, counts = redactor.redact(
        "For licensing questions please email us at contact@eleuther.ai."
    )
    assert "contact@eleuther.ai" not in text
    assert counts.get("EMAIL_ADDRESS") == 1


def test_cleaning_drops_inline_sponsor_credit_cta():
    blocks = [
        Block(type="paragraph", text=(
            "We will use Kaggle Notebooks because they work well for this course. "
            "(If you don't have a Paperspace account yet, sign up with this link to "
            "get $10 credit - and we get a credit too.) The course starts in January."
        )),
    ]
    doc = Document(id="cta-1", text="", url="https://course.fast.ai", blocks=blocks)
    cleaned = default_cleaning_chain(
        redact_pii=False,
        remove_references=True,
        remove_captions=True,
        remove_math=True,
        math_policy="text",
        table_policy="drop",
        code_policy="keep",
        caption_policy="drop",
    )(doc)
    assert "get $10 credit" not in cleaned.text
    assert "The course starts in January." in cleaned.text


def test_split_assignment_balances_by_size_and_keeps_documents_whole():
    # A corpus shaped like the real one: a few large documents plus a tail of
    # small ones.  Hash-only assignment would put a tiny document in val.
    sizes_in = [9102, 8831, 8280, 7085, 4468, 3808, 3459, 3196, 2840, 1954,
                1877, 1319, 1121, 988, 713, 326, 107]
    docs = [
        Document(id=f"doc-{i}", text=" ".join(["word"] * size), url=f"https://x/{i}", title="t")
        for i, size in enumerate(sizes_in)
    ]
    mapping = assign_splits(docs, {"train": 0.9, "val": 0.05, "test": 0.05})
    assert set(mapping) == {d.id for d in docs}
    assert set(mapping.values()) == {"train", "val", "test"}

    sizes = {d.id: len(d.text.split()) for d in docs}
    total = sum(sizes.values())
    for split in ("val", "test"):
        share = sum(sizes[doc_id] for doc_id, s in mapping.items() if s == split)
        # The eval splits must be a real share of the corpus, not one tiny doc.
        assert 0.02 * total <= share <= 0.12 * total, (split, share, total)
    train_share = sum(sizes[doc_id] for doc_id, s in mapping.items() if s == "train")
    assert train_share > 0.8 * total

    # Deterministic: the same corpus always produces the same assignment.
    assert assign_splits(docs, {"train": 0.9, "val": 0.05, "test": 0.05}) == mapping
