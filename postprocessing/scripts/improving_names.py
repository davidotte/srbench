import pandas as pd
import re


pretrained_algs = [
    'uDSR',
    'NeSymRes',
    'TPSR',
    'E2E',
    'FlashANSR',
    'RASR',
    'RASR Noise',
    'QLattice',
    'SNIP'
]

symbolic_algs = [
    'AFP', 
    'AFP_EHC',
    'AFP_FE',
    'Bingo',
    'Brush',
    'BSR',
    'E2E',
    'EPLEX',
    'EQL',
    'FEAT',
    'FFX',
    'FlashANSR',
    'RASR',
    'RASR Noise',
    'Genetic Engine',
    'Genetic Engine 1p1',
    'Genetic Engine hc',
    'Genetic Engine rs',
    'GP-GOMEA',
    'gplearn',
    'GPZGD',
    'ITEA', 
    # 'LightGBM', 
    'NeSymRes', 
    'Operon',
    'PS-Tree', 
    'PYSR',
    'QLattice',
    'Rils-Rols',
    'SNIP',
    'TIR',
    'TPSR',
    'uDSR'
]

nongp_algs = [
    'BSR',
    'DSR',
    'AIFeynman'
]

gp_algs = [ # These should have get_population method implemented
    'AFP', 
    'AFP_FE',
    'FFX',
    'FEAT',
    'EPLEX',
    'GP-GOMEA',
    'gplearn',
    'ITEA', 
    'TIR', 
    'PS-Tree',
    'MRGP', 
    'Operon',
    'SBP-GP',
]

def add_metadata(df):
    """Adds metadata regarding different classification of algorithms"""
    
    df['symbolic_alg'] = df['algorithm'].apply(lambda x: x in symbolic_algs)
    df['pre_trained'] = df['algorithm'].apply(lambda x: x in pretrained_algs)
    df['gp_alg'] = df['algorithm'].apply(lambda x: x in gp_algs)

    return df


def improve_names(df):
    """Will update the names with pretty formatting. All naming should be
    updated into this file to avoid mismatching names between different
    results analysis"""

    df['algorithm'] = df['algorithm'].astype(str)
    
    # remove regressor from names
    df['algorithm'] = df['algorithm'].apply(lambda x: x.replace('Regressor',''))
    # remove tuned from names
    df['algorithm'] = df['algorithm'].apply(lambda x: x.replace('tuned','')) 

    # Specific fixes
    name_map = {
        'afp' : 'AFP', 
        'afp_ehc' : 'AFP_EHC',
        'afp_fe' : 'AFP_FE',
        'bingo' : 'Bingo',
        'brush' : 'Brush',
        'bsr' : 'BSR',
        'e2et' : 'E2E',
        'eplex' : 'EPLEX',
        'eql' : 'EQL',
        'feat' : 'FEAT',
        'ffx' : 'FFX',
        'flashansr' : 'FlashANSR',
        'rasr' : 'RASR',
        'rasr_noise' : 'RASR Noise',
        'geneticengine' : 'Genetic Engine',
        'geneticengine_1p1' : 'Genetic Engine 1p1',
        'geneticengine_hc' : 'Genetic Engine hc',
        'geneticengine_rs' : 'Genetic Engine rs',
        'gpgomea' : 'GP-GOMEA',
        'gplearn' : 'gplearn',
        'gpzgd' : 'GPZGD',
        'itea' : 'ITEA', 
        'lightgbm' : 'LightGBM', 
        'nesymres' : 'NeSymRes', 
        'operon' : 'Operon',
        'ps-tree' : 'PS-Tree', 
        'pysr' : 'PYSR', 
        'qlattice' : 'QLattice', 
        'rils-rols' : 'Rils-Rols',
        'snip' : 'SNIP',
        'sklearn_adaboost' : 'adaboost',
        'sklearn_lasso' : 'Lasso',
        'sklearn_linear' : 'Linear',
        'sklearn_mlp' : 'MLP',
        'sklearn_randomforest' : 'RF',
        'sklearn_ridge' : 'Ridge',
        'sklearn_sgd' : 'SGD',
        'tir' : 'TIR',
        'tpsr' : 'TPSR', 
        'udsr' : 'uDSR',
        'xgboost' : 'XGB',
    }

    # Keep unknown/new algorithm names as-is (instead of turning into NaN).
    df['algorithm'] = df['algorithm'].map(name_map).fillna(df['algorithm'])

    return df