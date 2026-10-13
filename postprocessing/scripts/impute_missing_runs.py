"""Impute missing (algorithm, dataset, random_state) runs so failures count as bad results instead of being dropped."""
import os
from glob import glob

import numpy as np
import pandas as pd


def _linear_expression_size(n_features):
    # sympy node count of b + c0*x0 + ... : 3 nodes per term, plus Add and intercept
    return 3 * n_features + 2


def impute_missing_runs(
    df_results,
    df_sum,
    datadir=None,
    impute_r2_source='fixed',
    impute_r2_test=0.0,
    impute_model_size='dataset_median',
    impute_algorithms=None,
    linear_algorithm_name='Linear',
):
    """Return (df_results, df_sum) with missing runs imputed.

    impute_r2_source: 'fixed' (use impute_r2_test) or 'linear' (the Linear baseline's r2_test on that split).
    impute_model_size: 'dataset_median' (NaN if no successful seed), 'algorithm_median', 'algorithm_mean',
    'linear_expression_size' (needs datadir), or a fixed number.
    """
    algorithms = impute_algorithms if impute_algorithms is not None else df_results['algorithm'].unique()

    display_name = df_sum[['algorithm', '*algorithm*']].drop_duplicates().set_index('algorithm')['*algorithm*']

    model_size_by_algo = None
    model_size_by_algo_dataset = None
    if impute_model_size in ('algorithm_median', 'algorithm_mean'):
        agg = 'median' if impute_model_size == 'algorithm_median' else 'mean'
        model_size_by_algo = getattr(df_results.groupby('algorithm')['model_size'], agg)()
    elif impute_model_size == 'dataset_median':
        model_size_by_algo_dataset = df_results.groupby(['algorithm', 'dataset'])['model_size'].median()

    n_features_by_dataset = {}

    def n_features(dataset):
        if dataset not in n_features_by_dataset:
            path = glob(os.path.join(datadir, dataset, dataset + '*.tsv.gz'))[0]
            cols = pd.read_csv(path, sep='\t', nrows=0).columns
            n_features_by_dataset[dataset] = sum(1 for c in cols if c != 'target')
        return n_features_by_dataset[dataset]

    # var(y_test) per split from existing runs (mse / (1 - r2)), so imputed mse matches imputed r2
    valid_r2 = df_results['r2_test'] < 1 - 1e-9
    var_y_by_seed = (
        df_results.loc[valid_r2]
        .assign(_var_y=lambda d: d['mse_test'] / (1 - d['r2_test']))
        .groupby(['dataset', 'random_state'])['_var_y']
        .median()
    )

    added = []
    for dataset, ds_group in df_results.groupby('dataset'):
        random_states = ds_group['random_state'].unique()

        dataset_model_size = None
        if impute_model_size == 'linear_expression_size':
            dataset_model_size = _linear_expression_size(n_features(dataset))

        linear_r2_by_seed = None
        if impute_r2_source == 'linear':
            linear_r2_by_seed = ds_group.loc[
                ds_group['algorithm'] == linear_algorithm_name
            ].set_index('random_state')['r2_test']

        for algo in algorithms:
            present_states = set(ds_group.loc[ds_group['algorithm'] == algo, 'random_state'])
            missing_states = [rs for rs in random_states if rs not in present_states]
            if not missing_states:
                continue

            if impute_model_size == 'dataset_median':
                model_size = model_size_by_algo_dataset.get((algo, dataset), np.nan)
            elif impute_model_size in ('algorithm_median', 'algorithm_mean'):
                model_size = model_size_by_algo[algo]
            elif impute_model_size == 'linear_expression_size':
                model_size = dataset_model_size
            else:
                model_size = impute_model_size

            for rs in missing_states:
                if impute_r2_source == 'linear' and linear_r2_by_seed is not None and rs in linear_r2_by_seed.index:
                    r2_value = linear_r2_by_seed.loc[rs]
                else:
                    r2_value = impute_r2_test
                mse_value = var_y_by_seed.get((dataset, rs), np.nan) * (1 - r2_value)
                added.append({
                    'algorithm': algo,
                    'dataset': dataset,
                    'random_state': rs,
                    'r2_test': r2_value,
                    'r2_zero_test': max(r2_value, 0),
                    'model_size': model_size,
                    'mse_test': mse_value,
                    'mae_test': np.nan,
                    'training time (s)': np.nan,
                    'training time (hr)': np.nan,
                    'symbolic_model': None,
                    'params_str': '{}',
                    'friedman_dataset': '_fri_' in dataset,
                })

    if not added:
        print('No missing (algorithm, dataset, random_state) runs found to impute.')
        return df_results, df_sum

    df_missing = pd.DataFrame(added)
    missing_per_algo = df_missing.groupby('algorithm').size()
    print(f"Imputing {len(df_missing)} missing trial row(s) "
          f"(whole missing datasets and/or individual missing seeds) "
          f"across algorithms: {missing_per_algo.to_dict()}")
    df_results = pd.concat([df_results, df_missing], ignore_index=True)

    for col in [c for c in df_results.columns if c.endswith('test') or c.endswith('size')]:
        if col + '_rank_per_trial' not in df_results.columns:
            continue
        ascending = 'r2' not in col
        df_results[col + '_rank_per_trial'] = df_results \
            .groupby(['dataset', 'random_state'], group_keys=False)[col] \
            .apply(lambda x: round(x, 3).rank(ascending=ascending))

    # Rebuild df_sum as in collate_experiments_results.py
    df_sum = df_results.drop(columns=['symbolic_model', 'params_str'], errors='ignore') \
                        .groupby(['algorithm', 'dataset'], as_index=False).median(numeric_only=True)
    df_sum['rmse_test'] = df_sum['mse_test'].apply(np.sqrt)
    df_sum['log_mse_test'] = df_sum['mse_test'].apply(lambda x: np.log(1 + x))
    df_sum['*algorithm*'] = df_sum['algorithm'].map(display_name).fillna(df_sum['algorithm'])

    for col in [c for c in df_sum.columns if c.endswith('test') or c.endswith('size')]:
        ascending = 'r2' not in col
        df_sum[col + '_rank'] = df_sum.groupby('dataset')[col] \
            .transform(lambda x: round(x.rank(ascending=ascending), 3))
        df_sum[col + '_norm'] = df_sum.groupby('dataset')[col] \
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df_results, df_sum
