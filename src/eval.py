"""
src/eval.py — Shared evaluation helpers.

Provides two consistent F1 definitions across the project:
- token_overlap_f1: bag-of-tokens precision/recall/F1 (lenient, used by 03/05)
- span_f1: exact span match by start/end/type (strict, used by 04/06)

Plus coverage and a 4-query downstream usability test.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Token-overlap F1 (lenient — bag-of-tokens overlap)
# ----------------------------------------------------------------------------
_PUNCT = ".,;:()[]\"'"


def normalise_token(tok: str) -> str:
    return tok.lower().strip(_PUNCT)


def text_to_token_set(text: str | None) -> set[str]:
    """Convert an extraction string to a set of normalised tokens."""
    if not text:
        return set()
    if isinstance(text, list):
        text = " ".join(text)
    if text.strip().lower() in {"", "none", "null", "n/a", "not found"}:
        return set()
    parts = text.replace("|", " ").split()
    return {normalise_token(t) for t in parts if normalise_token(t)}


def token_overlap_f1(gold_tokens: set[str], pred_tokens: set[str]) -> tuple[float, float, float]:
    """
    Bag-of-tokens P/R/F1.

    Edge cases:
    - both empty -> (1, 1, 1) — correct null extraction
    - gold empty, pred not -> (0, 1, 0) — hallucination; recall undefined, set to 1 by convention
    - pred empty, gold not -> (0, 0, 0) — miss
    """
    if not gold_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    if not gold_tokens:
        return 0.0, 1.0, 0.0
    overlap = gold_tokens & pred_tokens
    p = len(overlap) / len(pred_tokens)
    r = len(overlap) / len(gold_tokens)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ----------------------------------------------------------------------------
# Span-level F1 (strict — exact start/end/type match)
# ----------------------------------------------------------------------------
def get_spans(tag_seqs: Sequence[Sequence[str]]) -> dict[str, list[tuple[int, int, int]]]:
    """Extract entity spans (start, end, sentence_id, type) from BIO sequences.

    Tags must be of the form B-X, I-X, or O. Returns a dict mapping the entity
    type to the list of (start, end, sentence_id) tuples.
    """
    spans: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for sid, seq in enumerate(tag_seqs):
        start, cur_type = -1, None
        for i, tag in enumerate(seq):
            if tag.startswith("B-"):
                if start >= 0:
                    spans[cur_type].append((start, i, sid))
                start, cur_type = i, tag[2:]
            elif tag.startswith("I-") and start >= 0 and tag[2:] == cur_type:
                pass
            else:
                if start >= 0:
                    spans[cur_type].append((start, i, sid))
                start, cur_type = -1, None
        if start >= 0:
            spans[cur_type].append((start, len(seq), sid))
    return dict(spans)


def span_f1(
    gold_seqs: Sequence[Sequence[str]],
    pred_seqs: Sequence[Sequence[str]],
    fields: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Strict span-level P/R/F1 per type and macro average."""
    gold_spans = get_spans(gold_seqs)
    pred_spans = get_spans(pred_seqs)
    if fields is None:
        fields = sorted(set(gold_spans) | set(pred_spans))

    scores: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for f in fields:
        g = set(gold_spans.get(f, []))
        p = set(pred_spans.get(f, []))
        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        scores[f] = {"precision": prec, "recall": rec, "f1": f1, "support": float(len(g))}
        f1s.append(f1)

    scores["macro"] = {
        "precision": float(np.mean([scores[f]["precision"] for f in fields])) if fields else 0.0,
        "recall": float(np.mean([scores[f]["recall"] for f in fields])) if fields else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }
    return scores


# ----------------------------------------------------------------------------
# Coverage
# ----------------------------------------------------------------------------
def coverage(predictions: dict[str, dict[str, str]], field: str) -> float:
    """Fraction of docs where a non-empty extraction was produced for `field`."""
    if not predictions:
        return 0.0
    n = len(predictions)
    covered = sum(1 for v in predictions.values() if v.get(field, "").strip())
    return covered / n if n else 0.0


# ----------------------------------------------------------------------------
# Downstream usability — keyword-style queries
# ----------------------------------------------------------------------------
DEFAULT_QUERIES = [
    ("interventions", "placebo",      "Trials with placebo"),
    ("participants",  "patient",      "Trials mentioning patients"),
    ("outcomes",      "pain",         "Trials measuring pain"),
    ("interventions", "rehabilitation", "Trials with rehabilitation"),
]


def run_queries(
    predictions: dict[str, dict[str, str]],
    queries=DEFAULT_QUERIES,
) -> list[dict]:
    """For each query (field, keyword), count how many docs have a match."""
    results = []
    n = len(predictions)
    for field, keyword, description in queries:
        matches = sum(
            1 for v in predictions.values()
            if keyword.lower() in v.get(field, "").lower()
        )
        results.append({
            "query": description,
            "field": field,
            "keyword": keyword,
            "matches": matches,
            "n_docs": n,
            "match_rate": matches / n if n else 0.0,
        })
    return results


# ----------------------------------------------------------------------------
# Save / load predictions
# ----------------------------------------------------------------------------
def save_predictions(approach: str, predictions: dict[str, dict[str, str]], pred_dir: str | Path = "./predictions") -> Path:
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{approach}.json"
    path.write_text(json.dumps(predictions, indent=2))
    return path


def load_predictions(approach: str, pred_dir: str | Path = "./predictions") -> dict[str, dict[str, str]]:
    path = Path(pred_dir) / f"{approach}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())
