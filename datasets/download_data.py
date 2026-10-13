import os
import urllib.request
import pmlb
from pmlb import fetch_data
import pandas as pd
import yaml

print(pmlb.__version__)

PMLB_GITHUB_URL = 'https://github.com/EpistasisLab/pmlb/raw/master/datasets'

ddir = '.'

print("Saving datasets to:", ddir)

dataset_categories = {
    'blackbox': ['1028_SWD', '1089_USCrime', '1193_BNG_lowbwt', '1199_BNG_echoMonths',
                 '192_vineyard', '210_cloud', '522_pm10', '557_analcatdata_apnea1',
                 '579_fri_c0_250_5', '606_fri_c2_1000_10', '650_fri_c0_500_50',
                 '678_visualizing_environmental'],
    'firstprinciples': ['first_principles_absorption', 'first_principles_bode',
                        'first_principles_hubble', 'first_principles_ideal_gas',
                        'first_principles_kepler', 'first_principles_leavitt',
                        'first_principles_newton', 'first_principles_planck',
                        'first_principles_rydberg', 'first_principles_schechter',
                        'first_principles_supernovae_zr',
                        'first_principles_tully_fisher'],
    'feynman': [
        'feynman_III_10_19', 'feynman_III_12_43', 'feynman_III_13_18', 'feynman_III_14_14',
        'feynman_III_15_12', 'feynman_III_15_14', 'feynman_III_15_27', 'feynman_III_17_37',
        'feynman_III_19_51', 'feynman_III_21_20', 'feynman_III_4_32', 'feynman_III_4_33',
        'feynman_III_7_38', 'feynman_III_8_54', 'feynman_III_9_52', 'feynman_II_10_9',
        'feynman_II_11_20', 'feynman_II_11_27', 'feynman_II_11_28', 'feynman_II_11_3',
        'feynman_II_13_17', 'feynman_II_13_23', 'feynman_II_13_34', 'feynman_II_15_4',
        'feynman_II_15_5', 'feynman_II_21_32', 'feynman_II_24_17', 'feynman_II_27_16',
        'feynman_II_27_18', 'feynman_II_2_42', 'feynman_II_34_11', 'feynman_II_34_2',
        'feynman_II_34_29a', 'feynman_II_34_29b', 'feynman_II_34_2a', 'feynman_II_35_18',
        'feynman_II_35_21', 'feynman_II_36_38', 'feynman_II_37_1', 'feynman_II_38_14',
        'feynman_II_38_3', 'feynman_II_3_24', 'feynman_II_4_23', 'feynman_II_6_11',
        'feynman_II_6_15a', 'feynman_II_6_15b', 'feynman_II_8_31', 'feynman_II_8_7',
        'feynman_I_10_7', 'feynman_I_11_19', 'feynman_I_12_1', 'feynman_I_12_11',
        'feynman_I_12_2', 'feynman_I_12_4', 'feynman_I_12_5', 'feynman_I_13_12',
        'feynman_I_13_4', 'feynman_I_14_3', 'feynman_I_14_4', 'feynman_I_15_10',
        'feynman_I_15_3t', 'feynman_I_15_3x', 'feynman_I_16_6', 'feynman_I_18_12',
        'feynman_I_18_14', 'feynman_I_18_4', 'feynman_I_24_6', 'feynman_I_25_13',
        'feynman_I_26_2', 'feynman_I_27_6', 'feynman_I_29_16', 'feynman_I_29_4',
        'feynman_I_30_3', 'feynman_I_30_5', 'feynman_I_32_17', 'feynman_I_32_5',
        'feynman_I_34_1', 'feynman_I_34_14', 'feynman_I_34_27', 'feynman_I_34_8',
        'feynman_I_37_4', 'feynman_I_38_12', 'feynman_I_39_1', 'feynman_I_39_11',
        'feynman_I_39_22', 'feynman_I_40_1', 'feynman_I_41_16', 'feynman_I_43_16',
        'feynman_I_43_31', 'feynman_I_43_43', 'feynman_I_44_4', 'feynman_I_47_23',
        'feynman_I_48_2', 'feynman_I_50_26', 'feynman_I_6_2', 'feynman_I_6_2a',
        'feynman_I_6_2b', 'feynman_I_8_14', 'feynman_I_9_18', 'feynman_test_1',
        'feynman_test_10', 'feynman_test_11', 'feynman_test_12', 'feynman_test_13',
        'feynman_test_14', 'feynman_test_15', 'feynman_test_16', 'feynman_test_17',
        'feynman_test_18', 'feynman_test_19', 'feynman_test_2', 'feynman_test_20',
        'feynman_test_3', 'feynman_test_4', 'feynman_test_5', 'feynman_test_6',
        'feynman_test_7', 'feynman_test_8', 'feynman_test_9',
    ],
    'strogatz': [
        'strogatz_bacres1', 'strogatz_bacres2', 'strogatz_barmag1', 'strogatz_barmag2',
        'strogatz_glider1', 'strogatz_glider2', 'strogatz_lv1', 'strogatz_lv2',
        'strogatz_predprey1', 'strogatz_predprey2', 'strogatz_shearflow1',
        'strogatz_shearflow2', 'strogatz_vdp1', 'strogatz_vdp2',
    ],
}

# categories whose ground-truth formula (needed for --sym_data symbolic-recovery
# evaluation) lives in the PMLB metadata.yaml's free-text 'description' field,
# e.g. "E_n = mom*sqrt(Bx**2+By**2+Bz**2)". These aren't in the installed pmlb
# package's dataset index, so they always go through the direct GitHub fallback.
SYM_DATA_CATEGORIES = {'feynman', 'strogatz'}


def fetch_true_description(dataset_name):
    """Fetch the 'description' field (containing the ground-truth formula)
    from the dataset's real metadata.yaml on PMLB's GitHub."""
    url = f'{PMLB_GITHUB_URL}/{dataset_name}/metadata.yaml'
    with urllib.request.urlopen(url) as f:
        remote_metadata = yaml.safe_load(f)
    return remote_metadata['description']

def fetch_dataset(dataset_name, category_path):
    """Fetch a dataset, falling back to direct GitHub download if not in installed PMLB."""
    dataset_path = os.path.join(category_path, dataset_name, dataset_name + '.tsv.gz')

    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path, sep='\t', compression='gzip')

    if dataset_name in pmlb.dataset_names:
        return fetch_data(dataset_name, local_cache_dir=category_path)

    # Dataset exists in PMLB GitHub but not in this installed package version
    url = f'{PMLB_GITHUB_URL}/{dataset_name}/{dataset_name}.tsv.gz'
    print(f"  Not in installed PMLB, downloading directly from {url}")
    dataset = pd.read_csv(url, sep='\t', compression='gzip')
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    dataset.to_csv(dataset_path, sep='\t', compression='gzip', index=False)
    return dataset

for category_name, dataset_list in dataset_categories.items():
    category_path = f"{ddir}/{category_name}"
    os.makedirs(category_path, exist_ok=True)

    for dataset_name in dataset_list:
        try:
            print(f"\nDownloading {dataset_name} to {category_name} category...")
            dataset = fetch_dataset(dataset_name, category_path)

            print(f"Successfully downloaded {dataset_name}")
        except Exception as e:
            print(f"Error downloading {dataset_name}: {str(e)}")
        
        # Write metadata.yml file
        metadata = {
            "name": dataset_name,
            "task": "regression",
            "n_instances": dataset.shape[0],
            "n_features": dataset.shape[1]
        }

        if category_name in SYM_DATA_CATEGORIES:
            # preserve the true-equation description needed by --sym_data
            # (experiment/symbolic_utils.py:get_sym_model)
            try:
                metadata["description"] = fetch_true_description(dataset_name)
            except Exception as e:
                print(f"Error fetching ground-truth description for {dataset_name}: {str(e)}")

        dataset_folder = os.path.join(category_path, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        metadata_path = os.path.join(dataset_folder, "metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)