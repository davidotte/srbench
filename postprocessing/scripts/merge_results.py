"""Merge new black-box JSON results into existing results.feather and results-summary.csv.gz."""

import os
import sys
import json
import shutil
import argparse
from glob import glob
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from improving_names import improve_names, add_metadata


def resolve_path(path_like):
    """Resolve a path relative to the current directory, falling back to the repo root."""
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_resolved = (Path.cwd() / candidate).resolve()
    if cwd_resolved.exists():
        return cwd_resolved

    repo_resolved = (REPO_ROOT / candidate).resolve()
    return repo_resolved


def apply_algorithm_renames(df, rename_map):
    """Rename algorithms case-insensitively."""
    if not rename_map:
        return df

    lower_map = {str(old).lower(): str(new) for old, new in rename_map.items()}
    df['algorithm'] = df['algorithm'].apply(
        lambda value: lower_map.get(str(value).lower(), value)
    )
    return df


def derive_train_output_dir(output_dir):
    output_path = Path(output_dir)
    return str(output_path.parent / f"{output_path.name}-with-train")


def merge_results(
    results_dir,
    output_dir,
    create_backup=True,
    rename_map=None,
    include_train_metrics=False,
):
    feather_path = os.path.join(output_dir, 'results.feather')
    summary_path = os.path.join(output_dir, 'results-summary.csv.gz')

    if not os.path.exists(feather_path):
        print(f"Warning: {feather_path} not found. Will create new file.")
        df_existing = pd.DataFrame()
    else:
        print(f"Loading existing results from {feather_path}...")
        df_existing = pd.read_feather(feather_path)
        print(f"  Existing: {len(df_existing)} rows, {df_existing['algorithm'].nunique()} algorithms")

    if create_backup and len(df_existing) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_feather = f"{feather_path}.backup_{timestamp}"
        backup_summary = f"{summary_path}.backup_{timestamp}" if os.path.exists(summary_path) else None

        shutil.copy(feather_path, backup_feather)
        print(f"  Backed up to {backup_feather}")

        if backup_summary:
            shutil.copy(summary_path, backup_summary)
            print(f"  Backed up to {backup_summary}")

    print(f"\nLoading new results from {results_dir}...")
    comparison_cols = [
        'dataset',
        'algorithm',
        'random_state',
        'time_time',
        'model_size',
        'symbolic_model',
        'r2_test',
        'mse_test',
        'mae_test',
        'params'
    ]
    if include_train_metrics:
        comparison_cols.extend(['r2_train', 'mse_train', 'mae_train'])

    frames = []
    json_files = glob(os.path.join(results_dir, '**/*.json'), recursive=True)

    if not json_files:
        json_files = glob(os.path.join(results_dir, '*/*.json'))

    print(f"  Found {len(json_files)} JSON files")

    for f in tqdm(json_files, desc="Loading JSON files"):
        # Ground-truth results are merged by merge_groundtruth_results.py
        if 'feynman_' in f or 'strogatz_' in f:
            continue

        try:
            r = json.load(open(f, 'r'))

            if 'cv_results' in f:
                r['algorithm'] = r['algorithm'].replace('tuned', '')

            if isinstance(r['symbolic_model'], list):
                sm = '+'.join([f'B{i}*{ri}' for i, ri in enumerate(r['symbolic_model'])])
                r['symbolic_model'] = sm

            frames.append({k: v for k, v in r.items() if k in comparison_cols})
        except Exception as e:
            print(f"  Warning: Error loading {f}: {e}")

    if not frames:
        print("  No valid results found to merge!")
        return

    print(f"  Loaded {len(frames)} new results")

    df_new = pd.DataFrame.from_records(frames)
    df_new['params_str'] = df_new.pop('params').apply(str)
    df_new = df_new.rename(columns={'time_time': 'training time (s)'})
    df_new['training time (hr)'] = df_new['training time (s)'] / 3600

    # Same transformations as collate_experiments_results.py
    print("  Applying name improvements and metadata...")
    df_new = improve_names(df_new)
    df_new = add_metadata(df_new)
    rename_map = resolve_bare_rename(rename_map, df_new)
    df_new = apply_algorithm_renames(df_new, rename_map)

    if rename_map:
        printable_map = ", ".join([f"{old}->{new}" for old, new in rename_map.items()])
        print(f"  Applied algorithm renames: {printable_map}")

    print("\nCombining results...")
    if len(df_existing) > 0:
        # New results replace existing ones with the same (dataset, algorithm, random_state)
        existing_keys = set(zip(df_existing['dataset'], df_existing['algorithm'], df_existing['random_state']))
        new_keys = set(zip(df_new['dataset'], df_new['algorithm'], df_new['random_state']))
        overlap = existing_keys & new_keys

        if overlap:
            print(f"  Warning: {len(overlap)} overlapping results found. New results will replace existing ones.")
            mask = ~df_existing.apply(
                lambda row: (row['dataset'], row['algorithm'], row['random_state']) in overlap,
                axis=1
            )
            df_existing = df_existing[mask]
            print(f"  Removed {len(mask) - mask.sum()} overlapping rows from existing data")

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    print("  Recalculating metrics...")
    df_combined['r2_zero_test'] = df_combined['r2_test'].apply(lambda x: max(x, 0))
    df_combined['friedman_dataset'] = df_combined['dataset'].str.contains('_fri_')

    # Assign in place to avoid a duplicate column if the feather already has it
    algorithm_counts = df_combined.groupby('dataset')['algorithm'].nunique()
    df_combined['algorithm_count'] = df_combined['dataset'].map(algorithm_counts)

    print("  Recalculating rankings...")
    rank_columns = [
        c for c in df_combined.columns
        if c.endswith('test') or c.endswith('size') or (include_train_metrics and c.endswith('train'))
    ]
    for col in rank_columns:
        ascending = 'r2' not in col
        df_combined[col + '_rank_per_trial'] = df_combined \
            .groupby(['dataset', 'random_state'], group_keys=False)[col] \
            .apply(lambda x: round(x, 3).rank(ascending=ascending))

    print(f"\nCombined results: {len(df_combined)} rows, {df_combined['algorithm'].nunique()} algorithms")
    print(f"  Algorithms: {sorted(df_combined['algorithm'].unique())}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving combined results to {feather_path}...")
    df_combined.to_feather(feather_path)

    # Summary: median over seeds, then per-dataset ranks and min-max norms
    print("Creating summary file...")
    df_sum = df_combined.drop(columns=['symbolic_model', 'params_str'], errors='ignore')\
                        .groupby(['algorithm', 'dataset'], as_index=False).median()

    df_sum['rmse_test'] = df_sum['mse_test'].apply(np.sqrt)
    df_sum['log_mse_test'] = df_sum['mse_test'].apply(lambda x: np.log(1 + x))
    if include_train_metrics and 'mse_train' in df_sum.columns:
        df_sum['rmse_train'] = df_sum['mse_train'].apply(np.sqrt)
        df_sum['log_mse_train'] = df_sum['mse_train'].apply(lambda x: np.log(1 + x))
    df_sum['*algorithm*'] = df_sum.apply(
        lambda row: ('*' if row['symbolic_alg'] else '') + row['algorithm'],
        axis=1
    )

    print("  Computing rankings and normalized scores...")
    summary_rank_columns = [
        c for c in df_sum.columns
        if c.endswith('test') or c.endswith('size') or (include_train_metrics and c.endswith('train'))
    ]
    for col in summary_rank_columns:
        ascending = 'r2' not in col

        df_sum[col + '_rank'] = df_sum.groupby('dataset')[col] \
            .transform(lambda x: round(x.rank(ascending=ascending), 3))

        df_sum[col + '_norm'] = df_sum.groupby('dataset')[col] \
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    print(f"  Summary: {len(df_sum)} rows, {df_sum['algorithm'].nunique()} algorithms")

    print(f"Saving summary to {summary_path}...")
    df_sum.to_csv(summary_path, compression='gzip', index=False)

    print("\n" + "="*60)
    print("Merge completed successfully!")
    print(f"  Results: {feather_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Total rows: {len(df_combined)}")
    print(f"  Total algorithms: {df_combined['algorithm'].nunique()}")
    print("="*60)


def parse_rename_pairs(rename_args):
    """Parse OLD:NEW or bare NEW values; a bare NEW is stored under the key None."""
    rename_map = {}
    for pair in rename_args:
        if ':' not in pair:
            new = pair.strip()
            if not new:
                raise ValueError(f"Invalid --rename-method value '{pair}'. Expected OLD:NEW or NEW.")
            rename_map[None] = new
            continue
        old, new = pair.split(':', 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"Invalid --rename-method value '{pair}'. Expected OLD:NEW.")
        rename_map[old] = new
    return rename_map


def resolve_bare_rename(rename_map, df_new):
    """Map a bare NEW rename onto the single incoming algorithm."""
    if None not in rename_map:
        return rename_map

    incoming_algorithms = sorted(df_new['algorithm'].unique())
    if len(incoming_algorithms) != 1:
        raise ValueError(
            "Bare --rename-method NEW requires exactly one incoming algorithm, "
            f"found {len(incoming_algorithms)}: {incoming_algorithms}. "
            "Use --rename-method OLD:NEW instead."
        )

    resolved = dict(rename_map)
    new_name = resolved.pop(None)
    resolved[incoming_algorithms[0]] = new_name
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Merge new JSON evaluation results into existing outputs.")
    parser.add_argument('results_directory', help='Directory containing JSON result files')
    parser.add_argument('output_directory', help='Directory containing existing results output files')
    parser.add_argument('--backup', action='store_true', help='Create backup of existing files before merge')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backups before merge')
    parser.add_argument(
        '--rename-method',
        action='append',
        default=[],
        metavar='OLD:NEW|NEW',
        help='Rename incoming algorithm OLD to NEW during merge (repeatable). '
             'A bare NEW renames the single incoming algorithm (assumes one method).',
    )
    parser.add_argument(
        '--output-directory-with-train',
        default=None,
        help='Optional explicit path for secondary output that keeps train metrics (default: <output_directory>-with-train).',
    )
    parser.add_argument(
        '--output-mode',
        choices=['both', 'without-train', 'with-train'],
        default='both',
        help='Choose which output(s) to create (default: both).',
    )

    args = parser.parse_args()

    results_dir = resolve_path(args.results_directory)
    output_dir = resolve_path(args.output_directory)
    create_backup = args.backup or not args.no_backup

    try:
        rename_map = parse_rename_pairs(args.rename_method)
    except ValueError as exc:
        parser.error(str(exc))

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    if args.output_mode in ('both', 'without-train'):
        merge_results(
            str(results_dir),
            str(output_dir),
            create_backup,
            rename_map,
            include_train_metrics=False,
        )

    if args.output_mode in ('both', 'with-train'):
        train_output_dir = args.output_directory_with_train or derive_train_output_dir(output_dir)
        print(f"\nCreating train-metric-inclusive output in: {train_output_dir}")
        merge_results(
            str(results_dir),
            str(train_output_dir),
            create_backup,
            rename_map,
            include_train_metrics=True,
        )


if __name__ == '__main__':
    main()
