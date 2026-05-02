"""
src/pico_table.py — Build the structured PICO table (the deliverable).

The whole point of the assessment is "convert unstructured abstracts into
structured tables". This module assembles the table from saved predictions.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .data import (
    DATA_DIR,
    FIELDS,
    get_abstract_text,
    get_gold_spans,
    load_eval_doc_ids,
)
from .eval import load_predictions


def build_pico_table(
    approach: str,
    eval_ids: list[str] | None = None,
    pred_dir: str | Path = "./predictions",
    include_gold: bool = True,
    include_abstract: bool = True,
    abstract_chars: int | None = None,
) -> pd.DataFrame:
    """Build a PICO dataframe from saved predictions for one approach.

    Columns:
        Document_ID, [Abstract], Participants, Interventions, Outcomes,
        [Gold_Participants, Gold_Interventions, Gold_Outcomes]

    Args:
        approach: name used when calling save_predictions(...)
        eval_ids: list of doc IDs to include. Defaults to the locked eval set.
        pred_dir: directory with prediction JSON files.
        include_gold: include the gold reference columns.
        include_abstract: include the abstract text.
        abstract_chars: optional character truncation for the abstract column.
    """
    if eval_ids is None:
        eval_ids = load_eval_doc_ids()

    preds = load_predictions(approach, pred_dir=pred_dir)
    if not preds:
        raise FileNotFoundError(
            f"No predictions found for approach '{approach}' in {pred_dir}. "
            "Make sure save_predictions() was called by the source notebook."
        )

    rows = []
    for doc_id in eval_ids:
        row = {"Document_ID": doc_id}
        if include_abstract:
            txt = get_abstract_text(doc_id)
            if abstract_chars and len(txt) > abstract_chars:
                txt = txt[:abstract_chars] + " ..."
            row["Abstract"] = txt

        pred_for_doc = preds.get(doc_id, {})
        for field in FIELDS:
            row[field.capitalize()] = pred_for_doc.get(field, "") or "Not Found"

        if include_gold:
            for field in FIELDS:
                gold = get_gold_spans(doc_id, field, split="test")
                row[f"Gold_{field.capitalize()}"] = " | ".join(gold) if gold else ""

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_combined_pico_table(
    approaches: list[str],
    eval_ids: list[str] | None = None,
    pred_dir: str | Path = "./predictions",
) -> pd.DataFrame:
    """Build a long-format PICO table comparing multiple approaches.

    Columns: Approach, Document_ID, Participants, Interventions, Outcomes,
             Gold_Participants, Gold_Interventions, Gold_Outcomes
    """
    frames = []
    for approach in approaches:
        try:
            df = build_pico_table(
                approach,
                eval_ids=eval_ids,
                pred_dir=pred_dir,
                include_abstract=False,
                include_gold=True,
            )
            df.insert(0, "Approach", approach)
            frames.append(df)
        except FileNotFoundError as e:
            print(f"Skipping {approach}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_pico_table(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
