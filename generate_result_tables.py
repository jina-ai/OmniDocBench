#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_result_tables.py

Enhanced script to discover multiple runs in a result folder, select runs,
and present comparison tables across runs.

Features:
- Automatic detection of runs by scanning files named like:
    <run>_<match_name>_*.json
  and extracting <run> as everything before "_<match_name>_".
- CLI options:
    --result-folder PATH   (default: ./result)
    --match-name NAME      (default: quick_match)
    --runs a,b,c           (comma-separated run names to show; default: auto-detect all)
    --ocr-types            alias for --runs (backwards compat)
    --show-all             print lots of per-category tables (default False)
- Constructs comparison tables:
    - Overall metrics (text_block_Edit_dist, display_formula_CDM, table_TEDS, ...)
    - text_block Edit distance by data_source (and mean)
    - reading_order breakdowns
    - page issues (fuzzy_scan, watermark, colorful_background)
    - text attributes and table attributes (numeric metrics flattened)
    - recognition-like metrics for text and tables (CER/WER/TEDS/accuracy)
    - repeat statistics (document-level): fraction of documents whose OCR response has
      repeated content at the end (bad generation); breakdowns by data source/language
- Uses pandas for pretty tabular output when available, otherwise prints JSON-like dicts.

This is a self-contained script intended to be run from the OmniDocBench directory:
    python3 generate_result_tables.py

Example:
    python3 generate_result_tables.py --result-folder ./result --match-name quick_match --runs step1-basic-25k-olmocr,step1-basic-25k-infinity

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional third-party libs
try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
except Exception:
    np = None  # type: ignore
    pd = None  # type: ignore


DEFAULT_RESULT_FOLDER = Path(__file__).resolve().parent / "result"
DEFAULT_MATCH_NAME = "quick_match"
OCR_TYPES_DICT_DEFAULT = {"end2end": "end2end"}


# -----------------------
# Utilities for JSON metric extraction
# -----------------------
def try_get_all_value(obj: Any) -> Optional[float]:
    """
    Given an object commonly found in metric JSON files, attempt to extract a single numeric
    aggregate value. Looks for common keys like 'ALL', 'ALL_page_avg', 'ALL_img_avg', 'value', 'score'.
    If obj is numeric, returns it as float. Otherwise returns None.
    """
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if not isinstance(obj, dict):
        return None
    for key in ("ALL", "ALL_page_avg", "ALL_img_avg", "value", "score", "mean", "avg"):
        if key in obj and isinstance(obj[key], (int, float)):
            return float(obj[key])
    # sometimes the whole dict contains numeric leaves directly (e.g. {"ALL": {"value": 0.5}})
    for v in obj.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None


def flatten_numeric_leaves(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """
    Recursively flatten numeric leaves from a nested dict into dotted keys.
    """
    out: Dict[str, float] = {}
    for k, v in d.items():
        key_name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)):
            out[key_name] = float(v)
        elif isinstance(v, dict):
            # first try to extract an aggregate from this dict
            val = try_get_all_value(v)
            if val is not None:
                out[key_name] = val
            else:
                deeper = flatten_numeric_leaves(v, key_name)
                out.update(deeper)
        # ignore lists for now
    return out


# -----------------------
# Discovery of runs and file helpers
# -----------------------
def discover_runs(result_folder: Path, match_name: str) -> List[str]:
    """
    Scan the result_folder for files matching "*_{match_name}_*.json".
    Extract run names as the substring before f'_{match_name}_'.
    Returns a sorted list of unique run names.
    """
    runs = set()
    pattern = f"*_{match_name}_*.json"
    for p in result_folder.glob(pattern):
        name = p.name
        split_token = f"_{match_name}_"
        if split_token in name:
            run = name.split(split_token, 1)[0]
            if run:
                runs.add(run)
    return sorted(runs)


def find_preferred_metric_file(result_folder: Path, run: str, match_name: str) -> Optional[Path]:
    """
    Given a run name, return the most likely 'metric result' JSON path.
    Preference order:
      - {run}_{match_name}_metric_result.json
      - {run}_{match_name}_result.json
      - any file starting with {run}_{match_name}_ and containing 'metric' or 'result'
      - None
    """
    candidates = [
        result_folder / f"{run}_{match_name}_metric_result.json",
        result_folder / f"{run}_{match_name}_result.json",
        result_folder / f"{run}_{match_name}_metric.json",
        result_folder / f"{run}_metric_result.json",  # recognition-only files (no match_name)
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: search for likely files
    for p in result_folder.glob(f"{run}_{match_name}_*.json"):
        if "metric" in p.name or "result" in p.name:
            return p
    return None


def load_json_safe(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------
# Build comparison tables
# -----------------------
def build_overall_map_from_result(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Build an 'overall map' for a single result dict. Keys mirror the original script:
    text_block_Edit_dist, display_formula_CDM, table_TEDS, table_TEDS_structure_only, reading_order_Edit_dist.
    CDM/TEDS-like values are converted to percentages (x100) where appropriate.
    """
    keys_spec: List[Tuple[str, str, bool]] = [
        ("text_block", "Edit_dist", False),
        ("display_formula", "CDM", True),
        ("table", "TEDS", True),
        ("table", "TEDS_structure_only", True),
        ("reading_order", "Edit_dist", False),
    ]
    out: Dict[str, float] = {}
    for cat, metric, is_pct in keys_spec:
        cat_obj = result.get(cat, {})
        value: Optional[float] = None
        if isinstance(cat_obj, dict):
            # try page-level metric first
            page_map = cat_obj.get("page", {})
            if isinstance(page_map, dict) and metric in page_map:
                value = try_get_all_value(page_map.get(metric))
                if value is None:
                    # attempt flatten
                    flattened = flatten_numeric_leaves(page_map.get(metric) or {})
                    if flattened:
                        # prefer keys that contain 'ALL'
                        chosen = None
                        for k in sorted(flattened.keys()):
                            if "ALL" in k:
                                chosen = flattened[k]
                                break
                        if chosen is None:
                            chosen = next(iter(flattened.values()))
                        value = chosen
            # try all-level
            if value is None:
                all_map = cat_obj.get("all", {})
                if isinstance(all_map, dict) and metric in all_map:
                    value = try_get_all_value(all_map.get(metric))
                    if value is None:
                        flattened = flatten_numeric_leaves(all_map.get(metric) or {})
                        if flattened:
                            value = next(iter(flattened.values()))
            # try top-level direct metric
            if value is None and metric in cat_obj:
                value = try_get_all_value(cat_obj.get(metric))
        if value is None:
            out[f"{cat}_{metric}"] = float("nan")
        else:
            out[f"{cat}_{metric}"] = float(value) * 100.0 if is_pct else float(value)
    # Document-level repeat ratio (fraction of docs with repeated content at the end)
    rs = result.get("repeat_stats", {}) or {}
    overall_repeat = rs.get("overall", {}) or {}
    repeat_val = overall_repeat.get("repeat_page_fraction") if isinstance(overall_repeat, dict) else None
    if repeat_val is not None and isinstance(repeat_val, (int, float)):
        out["repeat_ratio"] = float(repeat_val)
    else:
        out["repeat_ratio"] = float("nan")
    # Compute overall score matching the notebook formula:
    # ((1 - text_block_Edit_dist) * 100 + display_formula_CDM + table_TEDS) / 3
    out["overall"] = (
        (1.0 - out.get("text_block_Edit_dist", float("nan"))) * 100.0
        + out.get("display_formula_CDM", float("nan"))
        + out.get("table_TEDS", float("nan"))
    ) / 3.0
    return out


def collect_dict_for_runs(runs: Iterable[str], result_folder: Path, match_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the preferred metric/result JSON for each run and return mapping run -> parsed JSON dict.
    Skips runs whose files are missing or invalid.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        path = find_preferred_metric_file(result_folder, run, match_name)
        if path is None:
            print(f"Warning: no preferred metric file found for run '{run}' (skipping)", file=sys.stderr)
            continue
        try:
            out[run] = load_json_safe(path)
        except Exception as e:
            print(f"Warning: failed to load JSON for run '{run}' from {path}: {e}", file=sys.stderr)
            continue
    return out


def _extract_repeat_breakdown(breakdown: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten a per-category repeat breakdown dict into columns suitable for a DataFrame row.

    Repeat metric is document-level: fraction of documents with repeated content at the end.
    Each key in *breakdown* (e.g. a data_source or language name) produces two columns:
      <key>__frac  – document-level repeat ratio (fraction of documents with end-repetition)
      <key>__pages – number of pages/documents
    """
    entries: Dict[str, float] = {}
    for k, v in breakdown.items():
        if isinstance(v, dict):
            frac = float(v.get("repeat_page_fraction", 0.0))
            pages = int(v.get("pages", 0) or 0)
            entries[f"{k}__frac"] = frac
            entries[f"{k}__pages"] = float(pages)
        elif isinstance(v, (int, float)):
            # legacy numeric value – treat as frac; pages unknown
            entries[f"{k}__frac"] = float(v)
            entries[f"{k}__pages"] = 0.0
    return entries


def assemble_comparison_tables(results_by_run: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given mapping run -> result JSON dict, produce a collection of comparison tables (as DataFrames when possible).
    Returns a dict of named tables.
    """
    tables: Dict[str, Any] = {}
    # Overall metrics
    overall_rows: Dict[str, Dict[str, float]] = {}
    for run, res in results_by_run.items():
        overall_rows[run] = build_overall_map_from_result(res)
    if pd is not None:
        tables["overall"] = pd.DataFrame.from_dict(overall_rows, orient="index").round(4)
    else:
        tables["overall"] = overall_rows

    # text_block Edit_dist by data_source
    tb_datasource: Dict[str, Dict[str, float]] = {}
    for run, res in results_by_run.items():
        tbl = res.get("text_block", {}).get("page", {}).get("Edit_dist")
        if isinstance(tbl, dict):
            # convert numeric-like leaves to flattened mapping
            flattened = flatten_numeric_leaves(tbl)
            # If flattened returns e.g. 'data_source: book' keys, keep them; else keep direct dict values
            if flattened:
                # Rename 'ALL' key to 'mean' to match notebook output
                if "ALL" in flattened:
                    flattened["mean"] = flattened.pop("ALL")
                tb_datasource[run] = flattened
            else:
                # fallback: take numeric values directly from tbl
                tb_datasource[run] = {k: v for k, v in tbl.items() if isinstance(v, (int, float))}
    if pd is not None and tb_datasource:
        tables["text_block_data_source"] = pd.DataFrame.from_dict(tb_datasource, orient="index").round(4)
    else:
        tables["text_block_data_source"] = tb_datasource

    # reading_order per-layout Edit_dist (from page["Edit_dist"])
    ro_tables: Dict[str, Dict[str, float]] = {}
    for run, res in results_by_run.items():
        page_edit_dist = res.get("reading_order", {}).get("page", {}).get("Edit_dist", {})
        if isinstance(page_edit_dist, dict):
            entries = {k: float(v) for k, v in page_edit_dist.items() if isinstance(v, (int, float))}
            if entries:
                ro_tables[run] = entries
    if pd is not None and ro_tables:
        tables["reading_order"] = pd.DataFrame.from_dict(ro_tables, orient="index").round(4)
    else:
        tables["reading_order"] = ro_tables

    # page issues: collect fuzzy_scan, watermark, colorful_* from text_block page Edit_dist
    page_issues: Dict[str, Dict[str, float]] = {}
    candidates = ["fuzzy_scan", "watermark", "colorful_backgroud", "colorful_background"]
    for run, res in results_by_run.items():
        edit_dist_map = res.get("text_block", {}).get("page", {}).get("Edit_dist", {})
        if isinstance(edit_dist_map, dict):
            entries = {}
            for c in candidates:
                if c in edit_dist_map:
                    v = edit_dist_map[c]
                    if isinstance(v, (int, float)):
                        entries[c] = float(v)
                    elif isinstance(v, dict):
                        val = try_get_all_value(v)
                        if val is not None:
                            entries[c] = float(val)
            if entries:
                page_issues[run] = entries
    if pd is not None and page_issues:
        tables["page_issues"] = pd.DataFrame.from_dict(page_issues, orient="index").round(4)
    else:
        tables["page_issues"] = page_issues

    # Text attributes: result['text_block']['group']['Edit_dist'] (matches notebook Cell 5)
    text_attr_tables: Dict[str, Dict[str, float]] = {}
    for run, res in results_by_run.items():
        group_edit_dist = res.get("text_block", {}).get("group", {}).get("Edit_dist", {})
        if isinstance(group_edit_dist, dict):
            entries = {k: float(v) for k, v in group_edit_dist.items() if isinstance(v, (int, float))}
            if entries:
                text_attr_tables[run] = entries
    if pd is not None and text_attr_tables:
        tables["text_attributes"] = pd.DataFrame.from_dict(text_attr_tables, orient="index").round(4)
    else:
        tables["text_attributes"] = text_attr_tables

    # Table attributes: result['table']['group']['TEDS'] * 100 (matches notebook Cell 6)
    table_attr_tables: Dict[str, Dict[str, float]] = {}
    for run, res in results_by_run.items():
        group_teds = res.get("table", {}).get("group", {}).get("TEDS", {})
        if isinstance(group_teds, dict):
            entries = {k: float(v) * 100 for k, v in group_teds.items() if isinstance(v, (int, float))}
            if entries:
                table_attr_tables[run] = entries
    if pd is not None and table_attr_tables:
        tables["table_attributes"] = pd.DataFrame.from_dict(table_attr_tables, orient="index").round(4)
    else:
        tables["table_attributes"] = table_attr_tables

    # Recognition metrics for standalone OCR tools (matches notebook Cells 7 & 8).
    # Text recognition: top-level result["group"]["Edit_dist"] (standalone recognition file structure).
    # Table recognition: result["table"]["group"]["TEDS"] * 100.
    # These are populated when recognition-only result files (no match_name) are loaded.
    text_rec_tables: Dict[str, Dict[str, float]] = {}
    table_rec_tables: Dict[str, Dict[str, float]] = {}
    for run, res in results_by_run.items():
        # Text recognition: top-level group key (standalone OCR recognition file)
        top_group_edit_dist = res.get("group", {}).get("Edit_dist", {})
        if isinstance(top_group_edit_dist, dict):
            entries = {k: float(v) for k, v in top_group_edit_dist.items() if isinstance(v, (int, float))}
            if entries:
                text_rec_tables[run] = entries

        # Table recognition: result["table"]["group"]["TEDS"] * 100
        table_group_teds = res.get("table", {}).get("group", {}).get("TEDS", {})
        if isinstance(table_group_teds, dict):
            entries = {k: float(v) * 100 for k, v in table_group_teds.items() if isinstance(v, (int, float))}
            if entries:
                table_rec_tables[run] = entries

    if pd is not None and text_rec_tables:
        tables["text_recognition"] = pd.DataFrame.from_dict(text_rec_tables, orient="index").round(4)
    else:
        tables["text_recognition"] = text_rec_tables

    if pd is not None and table_rec_tables:
        tables["table_recognition"] = pd.DataFrame.from_dict(table_rec_tables, orient="index").round(4)
    else:
        tables["table_recognition"] = table_rec_tables

    # -----------------------
    # Repeat statistics extraction (document-level)
    # Metric: fraction of documents that have repeated content at the end of the OCR
    # response (model repeating tokens/phrases without properly ending).
    # Expect result['repeat_stats'] to contain:
    #   'overall': {'repeat_page_fraction', 'pages'}  (repeat_page_fraction = document-level repeat ratio)
    #   'by_data_source': { data_source: {'repeat_page_fraction', 'pages'}, ... }
    #   'by_language': { language: {...}, ... }
    # -----------------------
    repeat_overall_rows: Dict[str, Dict[str, float]] = {}
    repeat_by_ds: Dict[str, Dict[str, float]] = {}
    repeat_by_lang: Dict[str, Dict[str, float]] = {}

    for run, res in results_by_run.items():
        rs = res.get("repeat_stats", {}) or {}
        overall = rs.get("overall", {}) or {}
        if isinstance(overall, dict) and overall:
            # repeat_ratio = fraction of documents with repeated content at the end (document-level)
            repeat_overall_rows[run] = {
                "repeat_ratio": float(overall.get("repeat_page_fraction", 0.0)),
                "pages": int(overall.get("pages", 0) or 0),
            }

        by_ds = rs.get("by_data_source", {}) or {}
        if isinstance(by_ds, dict) and by_ds:
            entries = _extract_repeat_breakdown(by_ds)
            if entries:
                repeat_by_ds[run] = entries

        by_lang = rs.get("by_language", {}) or {}
        if isinstance(by_lang, dict) and by_lang:
            entries = _extract_repeat_breakdown(by_lang)
            if entries:
                repeat_by_lang[run] = entries

    if pd is not None and repeat_overall_rows:
        df = pd.DataFrame.from_dict(repeat_overall_rows, orient="index")
        # Keep 'pages' as integer, round the rest
        pages_col = df["pages"].astype(int) if "pages" in df.columns else None
        df = df.round(4)
        if pages_col is not None:
            df["pages"] = pages_col
        tables["repeat_overall"] = df
    else:
        tables["repeat_overall"] = repeat_overall_rows

    def _round_repeat_df(raw: Dict[str, Dict[str, float]]) -> Any:
        """Build a DataFrame from *raw*, rounding floats but keeping __pages columns as int."""
        if pd is None or not raw:
            return raw
        df = pd.DataFrame.from_dict(raw, orient="index")
        pages_cols = [c for c in df.columns if c.endswith("__pages")]
        df = df.round(4)
        for c in pages_cols:
            df[c] = df[c].astype(int)
        return df

    tables["repeat_by_data_source"] = _round_repeat_df(repeat_by_ds)
    tables["repeat_by_language"] = _round_repeat_df(repeat_by_lang)

    return tables


# -----------------------
# Pretty printing helpers
# -----------------------
def safe_print(obj: Any, title: Optional[str] = None) -> None:
    """
    Print a table as TSV (tab-separated values) for easy copy-paste into Excel / Google Sheets.
    """
    if title:
        print(f"\n== {title} ==\n")
    if pd is not None and isinstance(obj, pd.DataFrame):
        print(obj.to_csv(sep="\t", lineterminator="\n"))
    elif isinstance(obj, dict) and obj:
        # dict of dicts -> build DataFrame-like TSV
        first_val = next(iter(obj.values()))
        if isinstance(first_val, dict):
            all_cols: list[str] = []
            for v in obj.values():
                for k in v:
                    if k not in all_cols:
                        all_cols.append(k)
            header = "\t" + "\t".join(all_cols)
            print(header)
            for row_name, row_data in obj.items():
                vals = "\t".join(str(row_data.get(c, "")) for c in all_cols)
                print(f"{row_name}\t{vals}")
        else:
            print(json.dumps(obj, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(obj, indent=2, ensure_ascii=False))


def is_empty(obj: Any) -> bool:
    """
    Safe emptiness check that treats pandas DataFrame using .empty to avoid ambiguous truthiness.

    Returns True when the object is None, an empty DataFrame, or an empty container (dict/list/tuple/set/str).
    For other objects fall back to bool() but catch exceptions (e.g. ambiguous DataFrame truthness).
    """
    if obj is None:
        return True
    if pd is not None and isinstance(obj, pd.DataFrame):
        return obj.empty
    if isinstance(obj, (dict, list, tuple, set, str)):
        return len(obj) == 0
    try:
        return not bool(obj)
    except Exception:
        # If evaluating truthiness raises, treat it as not empty conservatively
        return False


# -----------------------
# CLI and main flow
# -----------------------
def parse_cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and compare result tables across runs.")
    p.add_argument("--result-folder", "-r", type=Path, default=DEFAULT_RESULT_FOLDER,
                   help="Folder containing result JSON files (default: ./result)")
    p.add_argument("--match-name", "-m", default=DEFAULT_MATCH_NAME,
                   help="Match name part of filenames (default: quick_match)")
    p.add_argument("--runs", "-R", default=None,
                   help="Comma-separated run names to show. If omitted, auto-detect all runs in result folder.")
    p.add_argument("--ocr-types", "-o", default=None,
                   help="Alias for --runs (backwards compatibility).")
    p.add_argument("--show-all", action="store_true", default=False,
                   help="Print all the extended tables (text/table attributes, recognition metrics).")
    p.add_argument("--limit-runs", type=int, default=0,
                   help="If >0, limit the number of runs displayed (helpful for many runs).")
    return p.parse_args()


def main() -> None:
    args = parse_cli_args()
    result_folder: Path = args.result_folder
    match_name: str = args.match_name
    runs_arg = args.runs or args.ocr_types

    if not result_folder.exists():
        print(f"Result folder does not exist: {result_folder}", file=sys.stderr)
        raise SystemExit(2)

    discovered = discover_runs(result_folder, match_name)
    if not discovered:
        print(f"No runs discovered in {result_folder} for match name '{match_name}'", file=sys.stderr)
        raise SystemExit(1)

    if runs_arg:
        requested = [r.strip() for r in runs_arg.split(",") if r.strip()]
        # validate requested runs against discovered
        runs = [r for r in requested if r in discovered]
        missing = [r for r in requested if r not in discovered]
        if missing:
            print(f"Warning: requested runs not found and will be ignored: {missing}", file=sys.stderr)
        if not runs:
            print("No valid runs to present after filtering; exiting.", file=sys.stderr)
            raise SystemExit(1)
    else:
        runs = discovered

    if args.limit_runs and args.limit_runs > 0:
        runs = runs[: args.limit_runs]

    print(f"Presenting runs for match_name='{match_name}': {runs}")

    results_by_run = collect_dict_for_runs(runs, result_folder, match_name)
    if not results_by_run:
        print("No results loaded for chosen runs (see warnings).", file=sys.stderr)
        raise SystemExit(1)

    tables = assemble_comparison_tables(results_by_run)

    # Print overall and main tables
    safe_keys = [
        ("overall", "OVERALL METRICS"),
        ("text_block_data_source", "TEXT_BLOCK: EDIT DIST BY DATA SOURCE"),
        ("reading_order", "READING ORDER (flattened)"),
        ("page_issues", "PAGE ISSUES (fuzzy_scan / watermark / colorful_backgroud)"),
        ("repeat_overall", "REPEAT AT END (document-level): repeat_ratio = fraction of docs with repeated content at end; pages"),
        ("repeat_by_data_source", "REPEAT AT END BY DATA SOURCE (frac = document-level repeat ratio, pages)"),
        ("repeat_by_language", "REPEAT AT END BY LANGUAGE (frac = document-level repeat ratio, pages)"),
    ]
    for key, title in safe_keys:
        table_obj = tables.get(key)
        # Use explicit emptiness check to avoid ambiguous DataFrame truth value
        if not is_empty(table_obj):
            safe_print(table_obj, title)

    # Print optional/extended tables if requested
    if args.show_all:
        extended_keys = [
            ("text_attributes", "TEXT ATTRIBUTES"),
            ("table_attributes", "TABLE ATTRIBUTES"),
            ("text_recognition", "TEXT RECOGNITION METRICS"),
            ("table_recognition", "TABLE RECOGNITION METRICS"),
        ]
        for key, title in extended_keys:
            table_obj = tables.get(key)
            # Use explicit emptiness check to avoid ambiguous DataFrame truth value
            if not is_empty(table_obj):
                safe_print(table_obj, title)

    # Also print a compact side-by-side summary (overall metrics transposed)
    overall = tables.get("overall")
    if overall is not None:
        if pd is not None and isinstance(overall, pd.DataFrame):
            safe_print(overall.T.round(4), "COMPARISON SUMMARY (overall metrics transposed)")
        else:
            safe_print(overall, "COMPARISON SUMMARY (overall metrics transposed)")


if __name__ == "__main__":
    main()
