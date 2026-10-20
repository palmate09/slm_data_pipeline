"""Offline tests for the SLM pipeline stages (no network required)."""

import os
import sys

# Ensure the repo root is on the path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from slm_pipeline.config import PipelineConfig
from slm_pipeline.dataset import DocumentDataset
from slm_pipeline.models import Document, new_id
from slm_pipeline.stage_a.extractor import TopicDocumentExtractor
from slm_pipeline.stage_b import DomainGate, build_default_ml_keywords
from slm_pipeline.stage_c import default_cleaning_chain
from slm_pipeline.stage_d import QualityFilter
from slm_pipeline.stage_e import ExactDuplicates, FuzzyDeduplicator

HTML_PAGE = """<html><head><title>Gradient Descent Explained</title></head><body>
<nav>Menu | Home | About</nav>
<main>
<p>Gradient descent is a first-order iterative optimization algorithm for finding
a local minimum of a differentiable function. In machine learning, we use
gradient descent to minimize the loss function of a neural network.</p>
<p>The update rule is: theta = theta - learning_rate * gradient. Repeating this
over many epochs trains the model by backpropagation of errors.</p>
</main>
<footer>Cookie policy | Subscribe to our newsletter | Related articles</footer>
</body></html>"""

OFFTOPIC_PAGE = """<html><head><title>Chocolate Chip Cookie Recipe</title></head><body>
<main>
<p>Preheat the oven to 350 degrees. Mix flour, sugar, butter and chocolate chips.
Bake for 12 minutes. These cookies are delicious with a glass of milk.</p>
</main>
</body></html>"""


def test_extractor_html():
    extractor = TopicDocumentExtractor()
    res = extractor.extract_text(text=HTML_PAGE, url="http://example.com/ml",
                                 doc_id="d1")
    assert res.ok is True
    assert "gradient descent" in res.document.text.lower()
    assert "backpropagation" in res.document.text.lower()
    assert "cookie policy" not in res.document.text.lower()
    assert "newsletter" not in res.document.text.lower()
    assert res.document.title == "Gradient Descent Explained"
    assert res.document.word_count > 20


def test_extractor_title_fallback():
    extractor = TopicDocumentExtractor()
    body = ("Neural networks are computing systems inspired by the biological "
            "neural networks that constitute animal brains. They learn to "
            "perform tasks by considering examples, generally without "
            "task-specific programming. A neural network is a collection of "
            "connected units called artificial neurons. These networks are "
            "the foundation of deep learning and are used across machine "
            "learning for classification, regression, and pattern recognition.")
    html = f"<html><body><p>{body}</p></body></html>"
    res = extractor.extract_text(text=html, url="http://example.com/x", doc_id="d2")
    assert res.ok is True, res.error
    assert res.document.title == ""
    assert res.document.word_count > 10


def test_extractor_empty():
    extractor = TopicDocumentExtractor()
    res = extractor.extract_text(text="<html><body></body></html>",
                                 url="http://example.com/empty", doc_id="d3")
    assert res.ok is False
    assert res.document.kept is False


def test_cleaning_chain():
    chain = default_cleaning_chain(fix_unicode=True, unify_quotes=True,
                                   redact_pii=True,
                                   pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"])
    doc = Document(id=new_id(), text=(
        "He said \u201chere is an email: alice@example.com and phone 555-123-4567\u201d"
        "\n\nCookie policy: we use cookies.\n\nMore text about \u201cneural networks\u201d."
    ), url="http://example.com", title="t")
    doc = chain(doc)
    assert "\u201c" not in doc.text and "\u201d" not in doc.text
    assert '"' in doc.text
    assert "alice@example.com" not in doc.text
    assert "555-123-4567" not in doc.text
    # A redacted identifier must not leak a marker into the training text, so
    # the sentence that carried it is dropped.
    assert "[REDACTED]" not in doc.text
    assert "cookie policy" not in doc.text.lower()
    assert "neural networks" in doc.text


def test_cleaning_mojibake():
    chain = default_cleaning_chain(fix_unicode=True, unify_quotes=True,
                                   redact_pii=False)
    # "MÃ¼nchen" is mojibake for "München" (latin-1 bytes read as utf-8).
    doc = Document(id=new_id(),
                   text="The MÃ¼nchen model uses a neural network.",
                   url="http://example.com", title="t")
    doc = chain(doc)
    assert "München" in doc.text, doc.text
    assert "neural network" in doc.text


def test_domain_gate_keyword_only():
    gate = DomainGate(
        topic="ml", keywords=["neural network", "gradient descent", "cooking"],
        keyword_min_hits=2, enable_embedding=False,
    )
    on = Document(id="1", text="This neural network uses gradient descent to learn.",
                  url="u", title="t")
    off = Document(id="2", text="Chocolate chip cookies are delicious to bake.",
                   url="u", title="t")
    assert gate.gate(on).keep is True
    assert gate.gate(off).keep is False
    assert gate.gate(off).layer == "keyword"


def test_domain_gate_default_keywords():
    kws = build_default_ml_keywords()
    assert "transformer" in kws and "pytorch" in kws
    gate = DomainGate(topic="ml", keywords=kws, keyword_min_hits=2,
                      enable_embedding=False)
    doc = Document(id="1", text="Backpropagation trains a transformer in PyTorch.",
                   url="u", title="t")
    assert gate.gate(doc).keep is True


def test_domain_gate_embedding_against_prototypes():
    gate = DomainGate(
        topic="ml",
        keywords=["machine learning", "neural network", "gradient descent"],
        keyword_min_hits=1,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        prototype_set_path="configs/ml_prototypes.txt",
        topic_similarity_threshold=0.3,
        enable_embedding=True,
    )
    # On-topic: mirrors the prototype definitions, passes keyword prefilter,
    # and should embed close to the prototype set.
    on = Document(id="1",
                  text=("A neural network is a series of algorithms that "
                        "recognizes relationships in data. Gradient descent is "
                        "an optimization algorithm for finding a local minimum "
                        "of a differentiable function. Backpropagation "
                        "calculates the gradient of the loss function with "
                        "respect to the weights of a neural network."),
                  url="u", title="t")
    # Off-topic semantically (baking), but contains an ML keyword so it passes
    # the cheap keyword prefilter and reaches the embedding layer.
    off = Document(id="2",
                   text=("I attended a neural network meetup to talk about "
                         "baking chocolate chip cookies on Sunday morning. "
                         "The oven should be preheated to 350 degrees."),
                   url="u", title="t")
    gr_on = gate.gate(on)
    gr_off = gate.gate(off)
    # Both must reach the embedding layer (contain >= 1 keyword hit).
    assert gr_on.layer == "embedding", gr_on
    assert gr_off.layer == "embedding", gr_off
    assert gr_on.topic_similarity is not None
    assert gr_off.topic_similarity is not None
    assert gr_on.topic_similarity > gr_off.topic_similarity
    assert gr_on.keep is True
    assert gr_off.keep is False


def test_quality_word_count():
    qf = QualityFilter(min_word_count=50, max_word_count=100000,
                       ngram_filters=[], max_perplexity=None)
    short = Document(id="1", text=" ".join(["word"] * 10), url="u", title="t")
    long_doc = Document(id="2", text=" ".join(["word"] * 200), url="u", title="t")
    assert qf(short).keep is False
    assert qf(long_doc).keep is True
    assert qf(long_doc).quality_score > 0


def test_cleaning_removes_wiki_artifacts():
    chain = default_cleaning_chain(fix_unicode=True, unify_quotes=True,
                                   redact_pii=False)
    doc = Document(id=new_id(), text=(
        "Optimization Notes\nOptimization Notes\n"
        "This article needs to be updated. Please help update this article.\n\n"
        "Gradient descent minimizes a loss function in machine learning.[12] "
        "It is used together with backpropagation.[citation needed]\n\n"
        "See also\n"
        "Gradient descent minimizes a loss function in machine learning."
    ), url="http://example.com", title="t")
    doc = chain(doc)
    lowered = doc.text.lower()
    assert "this article needs to be updated" not in lowered
    assert "please help update" not in lowered
    assert "[citation needed]" not in lowered
    assert "[12]" not in lowered
    assert "see also" not in lowered
    assert lowered.count("optimization notes") == 1
    assert "gradient descent minimizes a loss function" in lowered


def test_quality_rejects_excessive_urls():
    qf = QualityFilter(min_word_count=20, max_word_count=1000,
                       ngram_filters=[], max_perplexity=None)
    doc = Document(id="1", text=(
        "machine learning " * 40 + "https://example.com/page " * 20
    ), url="u", title="t")
    res = qf(doc)
    assert res.keep is False
    assert "URL" in res.reason


def test_quality_rejects_nonprintable_text():
    qf = QualityFilter(min_word_count=20, max_word_count=1000,
                       ngram_filters=[], max_perplexity=None)
    doc = Document(id="1", text=(
        "machine learning " * 40 + "\x00" * 20
    ), url="u", title="t")
    res = qf(doc)
    assert res.keep is False
    assert "non-printable" in res.reason


def test_quality_ngram_repetition():
    qf = QualityFilter(min_word_count=10, max_word_count=100000,
                       ngram_filters=[{"n": 2, "max_ratio": 0.3}],
                       max_perplexity=None)
    spam = Document(id="1", text="foo bar foo bar foo bar foo bar foo bar foo bar " * 20,
                    url="u", title="t")
    res = qf(spam)
    assert res.keep is False
    assert res.checks["top2gram_ratio"] > 0.3


def test_exact_dedup():
    dup = ExactDuplicates(text_field="text", id_field="id")
    docs = [
        Document(id="1", text="hello world neural network", url="u1"),
        Document(id="2", text="hello world neural network", url="u2"),
        Document(id="3", text="unique content here", url="u3"),
    ]
    res = dup(docs)
    assert res.exact_duplicates == 1
    assert docs[1].kept is False
    assert docs[0].kept is True
    assert docs[2].kept is True


def test_fuzzy_dedup():
    fuzz = FuzzyDeduplicator(threshold=0.5, num_perm=64)
    base = ("gradient descent optimizes the neural network weights "
            "by backpropagation through many layers")
    docs = [
        Document(id="1", text=base, url="u1"),
        Document(id="2", text=base + " and it works well.", url="u2"),  # near-dup
        Document(id="3", text="baking cookies in the oven is fun", url="u3"),
    ]
    res = fuzz(docs)
    assert res.fuzzy_duplicates >= 1, res.drop_reasons
    assert docs[2].kept is True


def test_dataset_jsonl_roundtrip(tmp_path):
    docs = [
        Document(id="1", text="hello world", url="u1", title="t1", quality_score=0.9),
        Document(id="2", text="foo bar", url="u2", title="t2", kept=False,
                 drop_reason="test"),
    ]
    ds = DocumentDataset(docs)
    p = tmp_path / "out.jsonl"
    ds.to_json(p)
    loaded = DocumentDataset.from_dicts([__import__("json").loads(l) for l in p.read_text().splitlines()])
    assert len(loaded) == 2
    assert loaded[0].text == "hello world"
    assert loaded[1].kept is False


def test_pipeline_config_save_load(tmp_path):
    cfg = PipelineConfig(topic="ml", min_word_count=30)
    p = tmp_path / "cfg.yaml"
    cfg.save(p)
    loaded = PipelineConfig.load(p)
    assert loaded.topic == "ml"
    assert loaded.min_word_count == 30
