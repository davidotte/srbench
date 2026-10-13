"""Remove one or more methods from results.feather and results-summary.csv.gz and recompute ranks."""

import os
import sys
import fnmatch
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def resolve_path(path_like):
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_resolved = (Path.cwd() / candidate).resolve()
    if cwd_resolved.exists():
        return cwd_resolved

    return (REPO_ROOT / candidate).resolve()


def derive_train_output_dir(output_dir):
    output_path = Path(output_dir)
    return str(output_path.parent / f"{output_path.name}-with-train")


def match_algorithms(pattern, algorithms):
    """Match with fnmatch if the pattern has wildcards, otherwise exactly."""
    if any(ch in pattern for ch in '*?['):
        return sorted(a for a in algorithms if fnmatch.fnmatch(a, pattern))
    return [pattern] if pattern in algorithms else []


def remove_method(output_dir, method_name, include_train_metrics=False):
    feather_path = os.path.join(output_dir, 'results.feather')
    summary_path = os.path.join(output_dir, 'results-summary.csv.gz')

    if not os.path.exists(feather_path):
        print(f"Error: {feather_path} not found.")
        return False

    print(f"Loading {feather_path}...")
    df = pd.read_feather(feather_path)
    print(f"  Loaded: {len(df)} rows, {df['algorithm'].nunique()} algorithms")

    matched = match_algorithms(method_name, df['algorithm'].unique())
    if not matched:
        print(f"  No algorithm matched '{method_name}'. No changes made.")
        print(f"  Available algorithms: {sorted(df['algorithm'].unique())}")
        return False

    print(f"  Matched {len(matched)} algorithm(s): {matched}")
    n_before = len(df)
    df = df[~df['algorithm'].isin(matched)].copy()
    print(f"  Removed {n_before - len(df)} rows")
    print(f"  Remaining: {len(df)} rows, {df['algorithm'].nunique()} algorithms")

    # Ground-truth results: group by noise level and data group so they aren't mixed
    noise_col = 'target_noise' if 'target_noise' in df.columns else None
    group_col = 'data_group' if 'data_group' in df.columns else None
    extra_group_cols = [c for c in [noise_col, group_col] if c]

    count_cols = ['dataset'] + ([noise_col] if noise_col else [])
    algorithm_counts = df.groupby(count_cols)['algorithm'].transform('nunique')
    df['algorithm_count'] = algorithm_counts

    print("  Recalculating per-trial rankings...")
    rank_columns = [
        c for c in df.columns
        if c.endswith('test') or c.endswith('size') or (include_train_metrics and c.endswith('train'))
    ]
    per_trial_group_cols = ['dataset', 'random_state'] + ([noise_col] if noise_col else [])
    for col in rank_columns:
        ascending = 'r2' not in col
        df[col + '_rank_per_trial'] = df \
            .groupby(per_trial_group_cols, group_keys=False)[col] \
            .apply(lambda x: round(x, 3).rank(ascending=ascending))

    print(f"Saving {feather_path}...")
    df.to_feather(feather_path)

    if not os.path.exists(summary_path):
        print(f"No pre-existing {summary_path}; skipping summary rebuild "
              "(this track doesn't use a summary file).")
        print(f"Done. Removed {matched} from {output_dir}.")
        return True

    print("Rebuilding summary...")
    df_sum = df.drop(columns=['symbolic_model', 'params_str'], errors='ignore') \
               .groupby(['algorithm', 'dataset'] + extra_group_cols, as_index=False).median()

    df_sum['rmse_test'] = df_sum['mse_test'].apply(np.sqrt)
    df_sum['log_mse_test'] = df_sum['mse_test'].apply(lambda x: np.log(1 + x))
    if include_train_metrics and 'mse_train' in df_sum.columns:
        df_sum['rmse_train'] = df_sum['mse_train'].apply(np.sqrt)
        df_sum['log_mse_train'] = df_sum['mse_train'].apply(lambda x: np.log(1 + x))
    if 'symbolic_alg' in df_sum.columns:
        df_sum['*algorithm*'] = df_sum.apply(
            lambda row: ('*' if row['symbolic_alg'] else '') + row['algorithm'],
            axis=1
        )

    summary_rank_columns = [
        c for c in df_sum.columns
        if c.endswith('test') or c.endswith('size') or (include_train_metrics and c.endswith('train'))
    ]
    summary_rank_group_cols = ['dataset'] + ([noise_col] if noise_col else [])
    for col in summary_rank_columns:
        ascending = 'r2' not in col
        df_sum[col + '_rank'] = df_sum.groupby(summary_rank_group_cols)[col] \
            .transform(lambda x: round(x.rank(ascending=ascending), 3))
        df_sum[col + '_norm'] = df_sum.groupby(summary_rank_group_cols)[col] \
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    print(f"Saving {summary_path}...")
    df_sum.to_csv(summary_path, compression='gzip', index=False)

    print(f"Done. Removed {matched} from {output_dir}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Remove a method from existing results output files.")
    parser.add_argument(
        'method_name',
        help='Algorithm name to remove (case-sensitive). Supports shell wildcards, '
             'e.g. "*oracle*" removes all algorithms containing "oracle".',
    )
    parser.add_argument(
        'output_directory',
        help='Directory containing results.feather (e.g. results/black-box or results/first-principles-tuning)',
    )
    parser.add_argument(
        '--output-directory-with-train',
        default=None,
        help='Optional explicit path for the with-train output directory.',
    )
    parser.add_argument(
        '--output-mode',
        choices=['both', 'without-train', 'with-train'],
        default='both',
        help='Choose which output(s) to update (default: both).',
    )

    args = parser.parse_args()
    output_dir = resolve_path(args.output_directory)

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    if args.output_mode in ('both', 'without-train'):
        print(f"\n--- Updating without-train output: {output_dir} ---")
        remove_method(str(output_dir), args.method_name, include_train_metrics=False)

    if args.output_mode in ('both', 'with-train'):
        train_dir = args.output_directory_with_train or derive_train_output_dir(output_dir)
        train_path = Path(train_dir)
        if not train_path.exists():
            print(f"\nSkipping with-train output (not found): {train_dir}")
        else:
            print(f"\n--- Updating with-train output: {train_dir} ---")
            remove_method(train_dir, args.method_name, include_train_metrics=True)


if __name__ == '__main__':
    main()
