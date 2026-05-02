"""
src/data.py — Shared data loading helpers for EBM-NLP.

Used across all notebooks so that:
- doc IDs, splits, and label conversion are defined once
- the evaluation set is identical everywhere
- the PICO CSV deliverable can be assembled from saved predictions

Usage from a notebook:
    import sys; sys.path.insert(0, '..')
    from src.data import (
        DATA_DIR, FIELDS,
        get_doc_ids, load_tokens, load_raw_labels, load_bio_labels,
        get_abstract_text, get_gold_spans, ensure_dataset,
        load_eval_doc_ids, save_eval_doc_ids,
    )
"""
from __future__ import annotations

import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# Default location relative to repo root. Each notebook can override
# by setting DATA_DIR before importing this module's helpers.
DATA_DIR = Path(os.environ.get("EBM_NLP_DIR", "./ebm_nlp_2_00"))
ARCHIVE_URL = "https://github.com/bepnye/EBM-NLP/raw/master/ebm_nlp_2_00.tar.gz"

DOCS_DIR = DATA_DIR / "documents"
ANNOT_DIR = DATA_DIR / "annotations" / "aggregated" / "hierarchical_labels"

FIELDS = ["participants", "interventions", "outcomes"]
TAG_MAP = {"participants": "P", "interventions": "I", "outcomes": "O"}

# Default location for the locked evaluation doc IDs file
EVAL_IDS_PATH = Path("./eval_doc_ids.json")


# ----------------------------------------------------------------------------
# Dataset bootstrap
# ----------------------------------------------------------------------------
def ensure_dataset(data_dir: Path = DATA_DIR, archive_path: str | Path = "./ebm_nlp_2_00.tar.gz") -> None:
    """Download + extract EBM-NLP if it isn't already present."""
    data_dir = Path(data_dir)
    archive_path = Path(archive_path)
    if data_dir.exists() and (data_dir / "documents").exists():
        return
    if not archive_path.exists():
        print(f"Downloading EBM-NLP from {ARCHIVE_URL} ...")
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
    print("Extracting archive ...")
    with tarfile.open(archive_path) as tf:
        tf.extractall(path=archive_path.parent)
    print(f"Dataset ready at {data_dir}")


# ----------------------------------------------------------------------------
# Doc IDs and raw IO
# ----------------------------------------------------------------------------
def _split_folder(split: str) -> str:
    return "test/gold" if split == "test" else split


def get_doc_ids(split: str = "train", label_type: str = "participants") -> list[str]:
    """Return sorted list of doc IDs for a given (split, field)."""
    folder = _split_folder(split)
    ann_dir = ANNOT_DIR / label_type / folder
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation dir missing: {ann_dir}")
    return sorted(p.stem.split(".")[0] for p in ann_dir.glob("*.AGGREGATED.ann"))


def load_tokens(doc_id: str) -> list[str] | None:
    path = DOCS_DIR / f"{doc_id}.tokens"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip().split("\n")


def load_raw_labels(doc_id: str, field: str, split: str = "train") -> list[int] | None:
    """Load raw hierarchical labels (integers 0..7) for one doc/field/split."""
    folder = _split_folder(split)
    path = ANNOT_DIR / field / folder / f"{doc_id}.AGGREGATED.ann"
    if not path.exists():
        return None
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(line))
        except ValueError:
            out.append(0)
    return out


def hierarchical_to_bio(raw: Iterable[int]) -> list[str]:
    """Flatten 0..7 hierarchical labels to BIO."""
    bio: list[str] = []
    prev = 0
    for t in raw:
        t = int(t)
        if t == 0:
            bio.append("O")
        else:
            bio.append("B" if prev == 0 else "I")
        prev = t
    return bio


def load_bio_labels(doc_id: str, field: str, split: str = "train") -> list[str] | None:
    raw = load_raw_labels(doc_id, field, split)
    return hierarchical_to_bio(raw) if raw is not None else None


# ----------------------------------------------------------------------------
# Convenience for evaluation
# ----------------------------------------------------------------------------
def get_abstract_text(doc_id: str) -> str:
    tokens = load_tokens(doc_id)
    return " ".join(tokens) if tokens else ""


def _spans_from_bio(tokens: list[str], bio: list[str]) -> list[str]:
    """Return list of entity surface strings from a BIO-tagged token sequence."""
    spans: list[str] = []
    cur: list[str] = []
    for tok, tag in zip(tokens, bio):
        if tag == "B":
            if cur:
                spans.append(" ".join(cur))
                cur = []
            cur.append(tok)
        elif tag == "I" and cur:
            cur.append(tok)
        else:
            if cur:
                spans.append(" ".join(cur))
                cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans


def get_gold_spans(doc_id: str, field: str, split: str = "test") -> list[str]:
    """Get the list of gold entity surface strings for a doc/field."""
    tokens = load_tokens(doc_id)
    bio = load_bio_labels(doc_id, field, split)
    if tokens is None or bio is None:
        return []
    return _spans_from_bio(tokens, bio)


# ----------------------------------------------------------------------------
# Locked evaluation set: every notebook should report numbers on the SAME set
# ----------------------------------------------------------------------------
def build_locked_eval_ids(n: int = 50, save: bool = True, path: Path = EVAL_IDS_PATH) -> list[str]:
    """Return the first `n` test doc IDs in P∩I∩O. Save to disk by default."""
    test_p = set(get_doc_ids("test", "participants"))
    test_i = set(get_doc_ids("test", "interventions"))
    test_o = set(get_doc_ids("test", "outcomes"))
    eval_ids = sorted(test_p & test_i & test_o)[:n]
    if save:
        Path(path).write_text(json.dumps(eval_ids, indent=2))
    return eval_ids


def load_eval_doc_ids(path: Path = EVAL_IDS_PATH) -> list[str]:
    path = Path(path)
    if not path.exists():
        return build_locked_eval_ids()
    return json.loads(path.read_text())


def save_eval_doc_ids(ids: list[str], path: Path = EVAL_IDS_PATH) -> None:
    Path(path).write_text(json.dumps(ids, indent=2))
