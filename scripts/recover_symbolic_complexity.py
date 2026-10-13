"""
Recover model_size for result rows where it is -1 by re-parsing the stored
symbolic_model strings with the same sympy pipeline as evaluate_model.py.

With --algorithm, rows of that algorithm whose symbolic_model contains a
literal `0.0` coefficient are re-checked as well, since sympy silently drops
such terms instead of failing.

Usage:
    python scripts/recover_symbolic_complexity.py [RESULTS_DIR ...] [--algorithm NAME] [--dry-run]
"""

import argparse
import re
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXPERIMENT_DIR = REPO_ROOT / "experiment"

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from symbolic_utils import complexity, get_sympy_model  # noqa: E402

DEFAULT_RESULT_DIRS = [REPO_ROOT / "results" / "black-box"]
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "blackbox"

UNRECOVERABLE_MODEL_STRINGS = {"not implemented", "nan", ""}


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


# Standalone zero float literal (`0.0`, `-0.0`, ...), not part of a larger number
ZERO_LITERAL_RE = re.compile(r"(?<![\d.])(-?)0\.0+(?!\d)")


def normalize_model_str(model_str):
    """Convert ellyn notation (`|x|`, `^`) to sympy syntax."""
    return model_str.replace("|", "").replace("^", "**")


def patch_zero_literals(model_str):
    """Replace each `0.0` literal by a distinct nonzero placeholder, so that
    terms whose coefficients were rounded to zero for display are kept.

    Returns (patched_str, n_patched).
    """
    counter = [0]

    def _sub(m):
        counter[0] += 1
        return f"{m.group(1)}0.00{counter[0]}1"

    patched = ZERO_LITERAL_RE.sub(_sub, model_str)
    return patched, counter[0]


def recover_model_size(model_str, dataset_path, timeout_s):
    """Recompute complexity from a stored symbolic_model string, retrying with
    patched zero literals if parsing fails.

    Returns (model_size, error, zero_literal_patched).
    """
    if not isinstance(model_str, str) or model_str.strip().lower() in UNRECOVERABLE_MODEL_STRINGS:
        return None, "no symbolic_model string available", False

    normalized = normalize_model_str(model_str)
    candidates = [(normalized, False)]
    patched, n_patched = patch_zero_literals(normalized)
    if n_patched > 0:
        candidates.append((patched, True))

    last_error = None
    for candidate, is_patched in candidates:
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_s)
        try:
            sympy_model = get_sympy_model(candidate, str(dataset_path))
            if not sympy_model.free_symbols:
                last_error = ("parsed to a feature-free constant "
                              f"({sympy_model}) - too much precision lost "
                              "in the stored string to recover model_size")
                continue
            model_size = int(complexity(sympy_model))
            return model_size, None, is_patched
        except _Timeout:
            last_error = f"sympy parsing exceeded {timeout_s}s timeout"
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return None, last_error, False


def dataset_path_for(dataset_dir, dataset_name):
    return dataset_dir / dataset_name / f"{dataset_name}.tsv.gz"


def _normalize_algorithm_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def recover_results_feather(df, dataset_dir, timeout_s, verbose=True,
                             feather_path=None, save_every=5, algorithm=None):
    """Recompute model_size for broken rows (model_size == -1, or with
    `algorithm` also rows containing zero literals). If `feather_path` is
    given, progress is saved every `save_every` rows.

    Returns the updated dataframe and a per-row report.
    """
    df = df.copy()
    df["model_size_recovered"] = False
    df["model_size_zero_literal_patched"] = False

    broken_mask = df["model_size"] == -1
    if algorithm is not None:
        target = _normalize_algorithm_name(algorithm)
        algo_mask = df["algorithm"].apply(_normalize_algorithm_name) == target
        if not algo_mask.any():
            available = sorted(df["algorithm"].unique())
            raise SystemExit(
                f"no rows match algorithm={algorithm!r}; available algorithms: {available}"
            )
        zero_literal_mask = df["symbolic_model"].fillna("").str.contains(ZERO_LITERAL_RE)
        broken_mask = algo_mask & (broken_mask | zero_literal_mask)

    broken_idx = df.index[broken_mask]
    total = len(broken_idx)

    if verbose:
        scope = f" for algorithm={algorithm!r}" if algorithm is not None else ""
        print(f"  {total} rows to (re)check{scope}")

    records = []
    for i, idx in enumerate(broken_idx, start=1):
        row = df.loc[idx]
        ds_path = dataset_path_for(dataset_dir, row["dataset"])
        if not ds_path.exists():
            error = f"dataset file not found: {ds_path}"
            changed = row["model_size"] != -1
            df.at[idx, "model_size"] = -1
            df.at[idx, "model_size_recovered"] = changed
            records.append((row["algorithm"], row["dataset"], False, changed, error, False))
        else:
            new_size, error, zero_literal_patched = recover_model_size(
                row["symbolic_model"], ds_path, timeout_s)
            if error is None:
                changed = new_size != row["model_size"]
                df.at[idx, "model_size"] = new_size
                df.at[idx, "model_size_recovered"] = changed
                df.at[idx, "model_size_zero_literal_patched"] = zero_literal_patched and changed
                records.append((row["algorithm"], row["dataset"], True, changed, None,
                                 zero_literal_patched and changed))
            else:
                changed = row["model_size"] != -1
                df.at[idx, "model_size"] = -1
                df.at[idx, "model_size_recovered"] = changed
                records.append((row["algorithm"], row["dataset"], False, changed, error, False))

        if verbose:
            recovered, changed = records[-1][2], records[-1][3]
            if recovered:
                status = "changed" if changed else "ok"
            else:
                status = "reset to -1" if changed else "failed (already -1)"
            print(f"  [{i}/{total}] {row['algorithm']} / {row['dataset']}: {status}")

        if feather_path is not None and i % save_every == 0:
            df.to_feather(feather_path)
            if verbose:
                print(f"    saved progress to {feather_path.name} ({i}/{total})")

    if feather_path is not None and total > 0:
        df.to_feather(feather_path)
        if verbose:
            print(f"    saved progress to {feather_path.name} ({total}/{total})")

    report = pd.DataFrame(
        records,
        columns=["algorithm", "dataset", "recovered", "changed", "error", "zero_literal_patched"],
    )

    # Same as postprocessing/scripts/collate_experiments_results.py
    if "random_state" in df.columns and "model_size_rank_per_trial" in df.columns:
        df["model_size_rank_per_trial"] = df.groupby(
            ["dataset", "random_state"], group_keys=False
        )["model_size"].transform(lambda x: round(x.rank(ascending=True), 3))

    return df, report


def refresh_summary_model_size(df_sum, df_results):
    """Recompute model_size, model_size_rank and model_size_norm in the summary
    as in postprocessing/scripts/collate_experiments_results.py."""
    df_sum = df_sum.copy()
    df_sum = df_sum.drop(columns=["model_size_recovered"], errors="ignore")

    median_size = (
        df_results.groupby(["algorithm", "dataset"])["model_size"]
        .median()
        .rename("model_size_new")
    )
    recovered_any = (
        df_results.groupby(["algorithm", "dataset"])["model_size_recovered"]
        .any()
        .rename("model_size_recovered")
    )

    df_sum = df_sum.merge(median_size, on=["algorithm", "dataset"], how="left")
    df_sum = df_sum.merge(recovered_any, on=["algorithm", "dataset"], how="left")
    df_sum["model_size_recovered"] = df_sum["model_size_recovered"].fillna(False)

    changed = df_sum["model_size_new"].notna() & (df_sum["model_size"] != df_sum["model_size_new"])
    df_sum.loc[changed, "model_size"] = df_sum.loc[changed, "model_size_new"]
    df_sum = df_sum.drop(columns=["model_size_new"])

    df_sum["model_size_rank"] = df_sum.groupby("dataset")["model_size"].transform(
        lambda x: round(x.rank(ascending=True), 3)
    )
    df_sum["model_size_norm"] = df_sum.groupby("dataset")["model_size"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    return df_sum, int(changed.sum())


def backup(path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.backup_{timestamp}")
    shutil.copy(path, backup_path)
    print(f"  backed up {path.name} -> {backup_path.name}")


def process_dir(results_dir, dataset_dir, timeout_s, do_backup, dry_run, algorithm=None):
    print(f"\n=== {results_dir} ===")
    feather_path = results_dir / "results.feather"
    summary_path = results_dir / "results-summary.csv.gz"

    if not feather_path.exists():
        print(f"  skipping: {feather_path} not found")
        return

    df_results = pd.read_feather(feather_path)

    if not dry_run and do_backup:
        backup(feather_path)

    df_results_fixed, report = recover_results_feather(
        df_results, dataset_dir, timeout_s,
        feather_path=None if dry_run else feather_path,
        algorithm=algorithm,
    )

    if len(report) == 0:
        print("  no candidate rows found, nothing to do")
        return

    summary = (
        report.groupby("algorithm")
        .agg(changed=("changed", "sum"), recovered=("recovered", "sum"), total=("recovered", "count"))
        .assign(still_failing=lambda d: d["total"] - d["recovered"])
    )
    print(summary.to_string())

    n_patched = int(report["zero_literal_patched"].sum())
    if n_patched:
        print(f"\n  {n_patched} row(s) changed by patching `0.0` coefficients")

    failures = report[~report["recovered"]]
    if len(failures) > 0:
        print(f"\n  {len(failures)} rows still unrecoverable, model_size set to -1, e.g.:")
        for _, row in failures.head(5).iterrows():
            print(f"    {row['algorithm']} / {row['dataset']}: {row['error']}")

    n_changed = int(report["changed"].sum())
    print(f"\n  changed model_size for {n_changed}/{len(report)} checked rows in results.feather")

    if dry_run:
        print("  --dry-run: not writing any files")
        return

    if n_changed == 0:
        print("  nothing changed, leaving files untouched")
        return

    print(f"  wrote {feather_path}")

    if summary_path.exists():
        df_sum = pd.read_csv(summary_path, compression="gzip")
        df_sum_fixed, n_summary_changed = refresh_summary_model_size(df_sum, df_results_fixed)
        if do_backup:
            backup(summary_path)
        df_sum_fixed.to_csv(summary_path, compression="gzip", index=False)
        print(f"  wrote {summary_path} ({n_summary_changed} algorithm/dataset entries updated)")


def main():
    parser = argparse.ArgumentParser(
        description="Recover broken model_size values by re-parsing stored symbolic_model strings.",
    )
    parser.add_argument(
        "results_dirs", nargs="*", type=Path,
        help="Result directories to process (default: results/black-box)",
    )
    parser.add_argument(
        "--algorithm", type=str, default=None,
        help="Restrict to one algorithm and also re-check rows with `0.0` coefficients.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--timeout", type=int, default=10, help="Per-expression timeout in seconds")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dirs = args.results_dirs or DEFAULT_RESULT_DIRS

    for results_dir in results_dirs:
        process_dir(
            results_dir=results_dir,
            dataset_dir=args.dataset_dir,
            timeout_s=args.timeout,
            do_backup=not args.no_backup,
            dry_run=args.dry_run,
            algorithm=args.algorithm,
        )


if __name__ == "__main__":
    main()
