"""Merge new ground-truth (Feynman/Strogatz) JSON results into an existing results.feather."""

import os
import sys
import json
import shutil
import argparse
from glob import glob
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from improving_names import improve_names, add_metadata
from merge_results import parse_rename_pairs, resolve_bare_rename, apply_algorithm_renames, resolve_path


def load_new_results(results_dir):
    """Load JSON results, preferring the .json.updated symbolic-assessment overlay when present."""
    frames, fails, bad_bsr = [], [], []
    updated = 0

    json_files = glob(os.path.join(results_dir, '*/*.json'))
    json_files_set = set(json_files)

    for f in tqdm(json_files, desc="Loading JSON files"):
        if 'EHC' in f:
            continue
        if f.endswith('_cv_results.json'):
            # Only a tuning intermediate if a final evaluate_model.py result exists
            plain_counterpart = f[: -len('_cv_results.json')] + '.json'
            if plain_counterpart in json_files_set:
                continue

        load_path = f
        if os.path.exists(f + '.updated'):
            load_path = f + '.updated'
            updated += 1
        try:
            r = json.load(open(load_path, 'r'))
            if isinstance(r['symbolic_model'], list):
                bad_bsr.append(load_path)
                sm = '+'.join([f'B{i}*{ri}' for i, ri in enumerate(r['symbolic_model'])])
                r['symbolic_model'] = sm
            r.pop('params', None)
            frames.append(r)
        except Exception as e:
            fails.append([load_path, e])

    print(f"  Found {len(json_files)} JSON files, {updated} with a symbolic-assessment overlay")
    if fails:
        print(f"  {len(fails)} failed to load:")
        for f in fails:
            print('   ', f[0], f[1])
    if bad_bsr:
        print('  Bad BSR models (list-valued symbolic_model):', bad_bsr)

    return frames


def finalize(df):
    """Recompute the derived columns of collate_groundtruth_results.py."""
    df['r2_zero_test'] = df['r2_test'].apply(lambda x: max(x, 0))
    df['data_group'] = df['dataset'].apply(lambda x: 'Feynman' if 'feynman' in x else 'Strogatz')

    # Runs not yet symbolically assessed lack these columns; treat them as not solved
    if 'simplified_symbolic_model' not in df.columns:
        df['simplified_symbolic_model'] = np.nan
    for col in ['symbolic_error_is_zero', 'symbolic_error_is_constant', 'symbolic_fraction_is_constant']:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False)

    df['symbolic_solution'] = df[[
        'symbolic_error_is_zero', 'symbolic_error_is_constant', 'symbolic_fraction_is_constant'
    ]].any(axis=1)
    df['symbolic_solution'] &= df['simplified_symbolic_model'].notna()
    df['symbolic_solution'] &= df['simplified_symbolic_model'] != '0'
    df['symbolic_solution'] &= df['simplified_symbolic_model'] != 'nan'

    # Rank per noise level, not only per dataset
    noise_col = 'target_noise' if 'target_noise' in df.columns else None
    group_cols = ['dataset', noise_col] if noise_col else ['dataset']
    for col, ascending in [('r2_test', False), ('model_size', True)]:
        df[col + '_rank'] = df.groupby(group_cols)[col] \
            .transform(lambda x: round(x.rank(ascending=ascending), 3))

    return df


def merge_groundtruth_results(results_dir, output_dir, create_backup=True, rename_map=None):
    feather_path = os.path.join(output_dir, 'results.feather')

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
        shutil.copy(feather_path, backup_feather)
        print(f"  Backed up to {backup_feather}")

    print(f"\nLoading new results from {results_dir}...")
    frames = load_new_results(results_dir)
    if not frames:
        print("  No valid results found to merge!")
        return

    print(f"  Loaded {len(frames)} new results")

    df_new = pd.DataFrame.from_records(frames)
    df_new = df_new.rename(columns={'time_time': 'training time (s)'})
    df_new['training time (hr)'] = df_new['training time (s)'] / 3600

    print("  Applying name improvements and metadata...")
    df_new = improve_names(df_new)
    df_new = add_metadata(df_new)
    rename_map = resolve_bare_rename(rename_map, df_new)
    df_new = apply_algorithm_renames(df_new, rename_map)

    if rename_map:
        printable_map = ", ".join([f"{old}->{new}" for old, new in rename_map.items()])
        print(f"  Applied algorithm renames: {printable_map}")

    print("\nCombining results...")
    # Noise levels are part of the key: the same (dataset, algorithm, seed) runs at several of them
    key_cols = ['dataset', 'algorithm', 'random_state', 'target_noise', 'feature_noise']
    if len(df_existing) > 0:
        existing_keys = set(zip(*[df_existing[c] for c in key_cols]))
        new_keys = set(zip(*[df_new[c] for c in key_cols]))
        overlap = existing_keys & new_keys

        if overlap:
            print(f"  Warning: {len(overlap)} overlapping results found. New results will replace existing ones.")
            mask = ~df_existing.apply(lambda row: tuple(row[c] for c in key_cols) in overlap, axis=1)
            df_existing = df_existing[mask]
            print(f"  Removed {len(mask) - mask.sum()} overlapping rows from existing data")

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    print("  Recalculating derived columns and rankings...")
    df_combined = finalize(df_combined)

    print(f"\nCombined results: {len(df_combined)} rows, {df_combined['algorithm'].nunique()} algorithms")
    print(f"  Algorithms: {sorted(df_combined['algorithm'].unique())}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving combined results to {feather_path}...")
    df_combined.to_feather(feather_path)

    print("\n" + "=" * 60)
    print("Merge completed successfully!")
    print(f"  Results: {feather_path}")
    print(f"  Total rows: {len(df_combined)}")
    print(f"  Total algorithms: {df_combined['algorithm'].nunique()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Merge new ground-truth (Feynman/Strogatz) JSON results into results/ground-truth/results.feather."
    )
    parser.add_argument('results_directory', help='Directory containing JSON result files')
    parser.add_argument('output_directory', help='Directory containing existing results.feather')
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

    merge_groundtruth_results(str(results_dir), str(output_dir), create_backup, rename_map)


if __name__ == '__main__':
    main()
