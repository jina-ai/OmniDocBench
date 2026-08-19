#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation entry point: builds an end2end_eval config from CLI args and runs the
same pipeline as `python pdf_validation.py --config configs/end2end.yaml`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import click

from src.core.pipeline import run_config

# Default metrics layout (same as configs/end2end.yaml)
DEFAULT_METRICS = {
    "text_block": {"metric": ["Edit_dist"]},
    "display_formula": {"metric": ["Edit_dist", "CDM"]},
    "table": {"metric": ["TEDS", "Edit_dist"]},
    "reading_order": {"metric": ["Edit_dist"]},
}


def build_config(
    gt_path: str,
    exp_path: str,
    truncate_repeats: bool,
    metrics: dict | None = None,
    match_method: str = "quick_match",
) -> dict:
    """Build the end2end_eval config dict from paths and options."""
    return {
        "end2end_eval": {
            "metrics": metrics if metrics is not None else DEFAULT_METRICS,
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": os.path.abspath(gt_path)},
                "prediction": {"data_path": os.path.abspath(exp_path)},
                "match_method": match_method,
                "truncated_repeats": truncate_repeats,
            },
        }
    }


@click.command()
@click.option(
    "--exp-path",
    "exp_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Result folder containing the prediction .md files.",
)
@click.option(
    "--gt-path",
    "gt_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to the ground-truth JSON file (e.g. OmniDocBench.json). Default: OmniDocBench.json next to this script.",
)
@click.option(
    "--match-method",
    "match_method",
    default="quick_match",
    type=click.Choice(["quick_match", "simple_match", "no_split"]),
    help="Matching method used to align ground truth with predictions.",
)
@click.option(
    "--truncate-repeats/--no-truncate-repeats",
    "truncate_repeats",
    default=True,
    help="Whether to truncate repeated content at the end before computing metrics.",
)
@click.option(
    "--metrics",
    "metrics_json",
    default=None,
    help=(
        "JSON string to override DEFAULT_METRICS, e.g. "
        '\'{"table": {"metric": ["TEDS"]}}\'. '
        "Merges into DEFAULT_METRICS (per-key override, not full replacement)."
    ),
)
def main(
    exp_path: Path,
    gt_path: Path | None,
    match_method: str,
    truncate_repeats: bool,
    metrics_json: str | None,
) -> None:
    """Run end-to-end evaluation: build a config from CLI args and run the evaluation pipeline.
    Results are written to ./result (default of the evaluation pipeline)."""
    script_dir = Path(__file__).resolve().parent
    if gt_path is None:
        gt_path = script_dir / "OmniDocBench.json"
    if not gt_path.exists():
        raise click.UsageError(
            f"Ground-truth file not found: {gt_path}. Pass --gt-path or place OmniDocBench.json next to evaluate.py."
        )
    gt_path = gt_path.resolve()

    metrics = None
    if metrics_json is not None:
        try:
            overrides = json.loads(metrics_json)
        except json.JSONDecodeError as e:
            raise click.UsageError(f"--metrics is not valid JSON: {e}") from e
        if not isinstance(overrides, dict):
            raise click.UsageError("--metrics must be a JSON object")
        metrics = {**DEFAULT_METRICS, **overrides}

    config = build_config(
        str(gt_path),
        str(exp_path.resolve()),
        truncate_repeats,
        metrics,
        match_method=match_method,
    )

    # Run with the repo root as cwd so `src` imports and ./result resolve correctly
    os.chdir(script_dir)
    run_config(config)


if __name__ == "__main__":
    main()
