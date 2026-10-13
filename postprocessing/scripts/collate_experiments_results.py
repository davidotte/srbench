"""
Collates JSON-formatted results, cleans them up, and saves them as .feather files.

Works with results generated with evaluate_model and optimize_model.
Does not handle results with varying levels of target noise.

For Feynman track experiments, see collate_groundtruth_results.py.
"""
# Original Author: William La Cava
# Modified by: Guilherme Aldeia
# License: GPLv3

# NOTE: you may need to install these: pyarrow, tqdm.
# It is better to install pyarrow via conda than pip.

import os
import sys
import json
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from improving_names import improve_names, add_metadata

# Directories
rdir = sys.argv[1] if len(sys.argv) > 1 else '../../results_blackbox/'
sdir = sys.argv[2] if len(sys.argv) > 2 else '../../results/black-box/'

print('Reading results from directory', rdir)
print('Saving summary to directory', sdir)

os.makedirs(sdir, exist_ok=True)

# Columns to keep
comparison_cols = [
    'dataset',
    'algorithm',
    'random_state',
    'time_time',
    'model_size',
    'symbolic_model',
    'r2_train',
    'r2_test',
    'mse_test',
    'mae_test',
    'params'
]

# Load JSON results
frames, fails = [], []
for f in tqdm(glob(rdir + '/*/*.json')):
    # leave out symbolic data
    if 'feynman_' in f or 'strogatz_' in f:
        continue
    try:
        r = json.load(open(f, 'r'))
        if 'cv_results' in f:
            # cleaning the "tuned" from the algorithm name
            r['algorithm'] = r['algorithm'].replace('tuned', '')

        if isinstance(r['symbolic_model'], list):
            sm = '+'.join([f'B{i}*{ri}' for i, ri in enumerate(r['symbolic_model'])])
            r['symbolic_model'] = sm

        frames.append({k: v for k, v in r.items() if k in comparison_cols})
    except Exception as e:
        fails.append([f, e])

print(len(fails), 'fails:', fails)

# Build DataFrame
df_results = pd.DataFrame.from_records(frames)
df_results['params_str'] = df_results.pop('params').apply(str)

# Cleanup
df_results = df_results.rename(columns={'time_time': 'training time (s)'})
df_results['training time (hr)'] = df_results['training time (s)'] / 3600

# Improve names and add metadata
df_results = improve_names(df_results)
df_results = add_metadata(df_results)

# Keep only symbolic regression methods
df_results = df_results[df_results["symbolic_alg"]]

print('Mean trial count per algorithm:')
print(df_results.groupby('algorithm')['dataset'].count() / df_results.dataset.nunique())

# Additional metrics
df_results['r2_zero_test'] = df_results['r2_test'].apply(lambda x: max(x, 0))
df_results['friedman_dataset'] = df_results['dataset'].str.contains('_fri_')
print('Loaded', len(df_results), 'results')

# Create summary
df_results2 = df_results.merge(
    df_results.groupby('dataset')['algorithm'].nunique().reset_index(),
    on='dataset', suffixes=('', '_count')
)

# Rankings per trial per dataset
for col in [c for c in df_results2.columns
            if c.endswith('test') or c.endswith('size') or c == 'r2_train']:
    ascending = 'r2' not in col
    df_results2[col + '_rank_per_trial'] = df_results2 \
        .groupby(['dataset', 'random_state'], group_keys=False)[col] \
        .apply(lambda x: round(x, 3).rank(ascending=ascending))

# Removing non-numerical columns first
df_sum = df_results2.drop(columns=['symbolic_model', 'params_str'])\
                    .groupby(['algorithm', 'dataset'], as_index=False).median()

df_sum['rmse_test'] = df_sum['mse_test'].apply(np.sqrt)
df_sum['log_mse_test'] = df_sum['mse_test'].apply(lambda x: np.log(1 + x))
df_sum['*algorithm*'] = df_sum.apply(lambda row: ('*' if row['symbolic_alg'] else '') + row['algorithm'], axis=1)

# Rankings and normalized scores
for col in [c for c in df_sum.columns
            if c.endswith('test') or c.endswith('size') or c == 'r2_train']:
    ascending = 'r2' not in col

    # Rank per dataset
    df_sum[col + '_rank'] = df_sum.groupby('dataset')[col] \
        .transform(lambda x: round(x.rank(ascending=ascending), 3))

    # Normalize per dataset
    df_sum[col + '_norm'] = df_sum.groupby('dataset')[col] \
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()))

for col in ['algorithm', 'dataset']:
    print(df_results[col].nunique(), col + 's')

# Quick glance at the results
avg_ranks = pd.DataFrame({
    'algorithm': df_sum['algorithm'].unique()
})

# Compute mean rank for r2_test (higher is better → ascending=False)
avg_ranks['r2_test_avg_rank'] = df_sum.groupby('algorithm')['r2_test_rank'] \
    .mean().reindex(avg_ranks['algorithm']).values

# Compute mean rank for model_size (smaller is better → ascending=True)
avg_ranks['model_size_avg_rank'] = df_sum.groupby('algorithm')['model_size_rank'] \
    .mean().reindex(avg_ranks['algorithm']).values

print("\nAverage ranks across datasets:")
print(avg_ranks.sort_values('r2_test_avg_rank'))

# Save results
df_sum.to_csv(f'{sdir}/results-summary.csv.gz', compression='gzip', index=False)
df_results.to_feather(f'{sdir}/results.feather')

print(f'Summary saved to {sdir}/results-summary.csv.gz')
print(f'Results saved to {sdir}/results.feather')
