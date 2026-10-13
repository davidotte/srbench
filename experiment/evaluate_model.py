import sys
import itertools
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
from sklearn.model_selection import train_test_split
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
from read_file import read_file
import pdb
import numpy as np
import json
import re
import os
import eco2ai

import sympy as sp
import inspect
from utils import jsonify
from symbolic_utils import complexity, get_sympy_model
from symbolic_utils import get_sym_model

from metrics.evaluation import simplicity

import signal
class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print(f"raising TimeOutException")
    raise TimeOutException

def set_env_vars(n_jobs):
    os.environ['OMP_NUM_THREADS'] = n_jobs 
    os.environ['OPENBLAS_NUM_THREADS'] = n_jobs 
    os.environ['MKL_NUM_THREADS'] = n_jobs

def evaluate_model(*, 
    # minimal working experiment
    dataset, 
    results_path,
    random_state,
    est_name,
    est,
    model,
    algorithm,

    # Extra configurations
    ecotracker=False,
    test=False,
    target_noise=0.0, 
    feature_noise=0.0, 
    use_tuned=False,
    fit_time_limit=3600,

    # valid options for eval_kwargs (may be specific for some algorithms, so they can set 
    # and it will be overriden here)
    sym_data=False,
    save_pop=False,
    test_params={},
    max_train_samples=0,
    scale_x=True,
    scale_y=True,
    pre_train=None,
    use_dataframe=True
):

    print(40*'=','Evaluating '+est_name+' on ',dataset,40*'=',sep='\n')

    np.random.seed(random_state)
    if hasattr(est, 'random_state'):
        est.random_state = random_state

    ##################################################
    # setup data
    ##################################################
    if ("e2et" in est_name) or ('tpsr' in est_name) or ('nesymres' in est_name) \
    or ("dso" in est_name) or ('bingo' in est_name):
        use_dataframe = False
    
    ##################################################
    # setup data
    ##################################################
    features, labels, feature_names = read_file(
        dataset, 
        use_dataframe=use_dataframe
    )
    print('feature_names:',feature_names)
    
    if sym_data:
        true_model = get_sym_model(dataset)

    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_state)

    # if dataset is large, subsample the training set 
    if max_train_samples > 0 and len(y_train) > max_train_samples:
        print('subsampling training data from',len(X_train),
              'to',max_train_samples)
        sample_idx = np.random.choice(np.arange(len(X_train)),
                                      size=max_train_samples,
                                      replace=False)
        y_train = y_train[sample_idx]
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.iloc[sample_idx]
        else:
            X_train = X_train[sample_idx]

    # scale and normalize the data
    if scale_x:
        print('scaling X')
        sc_X = StandardScaler() 
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
        if use_dataframe:
            X_train_scaled = pd.DataFrame(X_train_scaled, 
                                          columns=feature_names)
            X_test_scaled = pd.DataFrame(X_test_scaled, 
                                          columns=feature_names)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if scale_y:
        print('scaling y')
        sc_y = StandardScaler()
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
    else:
        y_train_scaled = y_train

    ################################################## 
    # noise
    ################################################## 
    if target_noise > 0:
        print('adding',target_noise,'noise to target')
        y_train_scaled += np.random.normal(0, 
                    target_noise*np.sqrt(np.mean(np.square(y_train_scaled))),
                    size=len(y_train_scaled))
    # add noise to the features
    if feature_noise > 0:
        print('adding',target_noise,'noise to features')
        X_train_scaled = np.array([x 
            + np.random.normal(0, feature_noise*np.sqrt(np.mean(np.square(x))),
                               size=len(x))
                                   for x in X_train_scaled.T]).T

    ################################################## 
    # run any method-specific pre_train routines
    ################################################## 
    if pre_train:
        pre_train(est, X_train_scaled, y_train_scaled)

    # define a test mode using estimator test_params, if they exist
    if test and len(test_params) != 0:
        est.set_params(**test_params)

    if use_tuned:
        try:
            tuned = importlib.__import__('methods.'+est_name+'._params',
                                        globals(), locals(), ['*'] )
            est.set_params(**tuned.params)
        except Exception as e:
            print(f"Tried to use tuned version of algorithm {est_name}, "+
                   "but no hyperparameter tuning step was performed on this "+
                   "algorithm yet. run optimize_model first.")
        est_name = 'tuned'+est_name

    ################################################## 
    # Fit models
    ################################################## 
    id = '_'.join([dataset.split('/')[-1].split('.')[0],
                      est_name,
                      str(random_state),
                      ])

    if target_noise > 0:
        id += '_target-noise'+str(target_noise)
    if feature_noise > 0:
        id += '_feature-noise'+str(feature_noise)

    if ecotracker:
        # file name should be something that will avoid parallel writing
        tracker = eco2ai.Tracker(
            project_name=dataset.split('/')[-1].split('.')[0], # dataset
            experiment_description=f'{est_name} {random_state}', # ml method and random seed
            file_name=os.path.join(results_path,id+"_eco2ai.csv"),
            alpha_2_code='US'
        )

    if not use_dataframe: 
        assert isinstance(X_train_scaled, np.ndarray)
        assert isinstance(X_test_scaled, np.ndarray)

    print('X_train:',type(X_train_scaled),X_train_scaled.shape)
    print('y_train:',y_train_scaled.shape)
    print('training',est)
    
    # time limits
    MAXTIME = fit_time_limit
    if hasattr(est, 'max_time'):
        est.max_time = MAXTIME
        print('max time set:',MAXTIME)
    elif hasattr(est, 'timeout_in_seconds'): # pysr
        est.timeout_in_seconds = MAXTIME
        print('max time set:',MAXTIME)
    elif hasattr(est, 'timeout'): # gpzgd
        est.timeout = MAXTIME
        print('max time set:',MAXTIME)
    elif hasattr(est, 'stop_time'): # nesymres
        est.stop_time = MAXTIME
        print('max time set:',MAXTIME)
    elif hasattr(est, 'time_limit'): # afp, afp_fe, afp_ehc, eplex
        est.time_limit = MAXTIME
        print('max time set:',MAXTIME)
    else:
        print('max time not set. Program will be killed if execution takes too long')

    if ecotracker:
        tracker.start()

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAXTIME + 600) # maximum time with some extra juice for finishing up

    t0t = time.time()
    try:
        est.fit(X_train_scaled, y_train_scaled)
        # if hasattr(est, 'oracle_select'):
        #     y_test_oracle = sc_y.transform(y_test.reshape(-1, 1)).flatten() if scale_y else y_test
        #     est.oracle_select(X_test_scaled, y_test_oracle)
    except TimeOutException:
        print("="*80)
        print('WARNING: fitting timed out. If the program does not handle SIGALRM, then it will be killed')
        print("="*80)
    finally:
        signal.alarm(0)  # Cancel the alarm
    time_time = time.time() - t0t
    if ecotracker:
        tracker.stop()
    print('Training time measure:', time_time)

    if 'geneticengine' in est_name:
        est._is_fitted = True

    ##################################################
    # store results
    ##################################################
    dataset_name = dataset.split('/')[-1].split('.')[0]
    results = {
        'dataset':dataset_name,
        'algorithm':est_name,
        'params':jsonify(est.get_params()),
        'random_state':random_state,
        'time_time': time_time, 
    }
    if sym_data:
        results['true_model'] = true_model

    # get the final symbolic model as a string
    print('fitted est:',est)

    if model is None:
        results['symbolic_model'] = "not implemented"
    elif 'X' in inspect.signature(model).parameters.keys():
        if not isinstance(X_train_scaled, pd.DataFrame):
            X_df = pd.DataFrame(X_train_scaled, 
                                          columns=feature_names)
        else:
            X_df = X_train_scaled
        results['symbolic_model'] = model(est, X_df)
    else:
        results['symbolic_model'] = model(est)

    print('symbolic model:',results['symbolic_model'])

    ##################################################
    # scores
    ##################################################
    for fold, target, X in  [ 
                             ['train', y_train, X_train_scaled], 
                             ['test', y_test, X_test_scaled]
                            ]:

        y_pred = np.asarray(est.predict(X)).reshape(-1,1)
        if scale_y:
            y_pred = sc_y.inverse_transform(y_pred)

        for score, scorer in [('mse',mean_squared_error),
                              ('mae',mean_absolute_error),
                              ('r2', r2_score)
                             ]:
            results[score + '_' + fold] = scorer(target, y_pred) 
    
    # simplicity
    if results['symbolic_model'] != "not implemented":
        results['simplicity'] = simplicity(results['symbolic_model'], feature_names)
    else:
        results['simplicity'] = np.nan

    def sympy_complexity(est):
        sympy_str = None
        if model is None:
            sympy_str = "not implemented"
        elif 'X' in inspect.signature(model).parameters.keys():
            if not isinstance(X_train_scaled, pd.DataFrame):
                X_df = pd.DataFrame(X_train_scaled, 
                                            columns=feature_names)
            else:
                X_df = X_train_scaled
            sympy_str = model(est, X_df)
        else:
            sympy_str = model(est)

        c = -1
        try:
            c = complexity(get_sympy_model(sympy_str, dataset))
        except:
            print(f"{est_name} does not have a complexity() method, and does not"
                " generate sympy-compatible expressions. setting to -1")
            return -1

        return int(c)
            
    # Forcing all algorithms to use same notion of complexity
    cplx = sympy_complexity(est)
    results['complexity_function'] = 'sympy'

    # if sympy fails we will use their methods. This should be deprecated eventually
    if cplx == -1 and ('complexity' in dir(algorithm) and algorithm.complexity is not None): 
        cplx = algorithm.complexity(est)
        results['complexity_function'] = 'user_defined'
    
    results['model_size'] = cplx
    results['target_noise']  = target_noise
    results['feature_noise'] = feature_noise

    ##################################################
    # Population analysis
    ##################################################
    if 'get_population' in dir(algorithm) and save_pop:
        population = algorithm.get_population(algorithm.est)
        population = population[:np.minimum(100, len(population))]

        frames = []
        for i, p in enumerate(population):
            frame = {
                'dataset': dataset_name,
                'algorithm': est_name,
                'random_state':random_state,
                "index" : i,
                "model_str" : algorithm.model(p),
                "model_size" : algorithm.complexity(p)
            }
            for fold, target, X in  [ 
                ['train', y_train, X_train_scaled], 
                ['test', y_test, X_test_scaled]
            ]:
                y_pred = None
                if 'gplearn'== est_name:
                    y_pred = np.asarray(p.execute(X.values)).reshape(-1,1)
                else:
                    y_pred = np.asarray(p.predict(X)).reshape(-1,1)
                
                if scale_y:
                    y_pred = sc_y.inverse_transform(y_pred)

                for score, scorer in [('mse',mean_squared_error),
                    ('mae',mean_absolute_error),
                    ('r2', r2_score)
                ]:
                    frame[f'{score}_{fold}'] = scorer(target, y_pred) 
            frames.append(frame)

        pd.DataFrame.from_records(frames).to_csv(
            os.path.join(results_path,id+"_population.csv"), index=False)

    ##################################################
    # write to file
    ##################################################
    print('results:')
    print(json.dumps(results, indent=4))
    print('---')

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file = os.path.join(
        results_path,
        '_'.join([dataset_name, est_name, str(random_state)])
    )

    if target_noise > 0:
        save_file += '_target-noise'+str(target_noise)
    if feature_noise > 0:
        save_file += '_feature-noise'+str(feature_noise)
        
    print('save_file:',save_file)

    with open(save_file + '.json', 'w') as out:
        json.dump(jsonify(results), out, indent=4)

    return save_file + '.json'

################################################################################
# main entry point
################################################################################
import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')     
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG',default=None,type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument(
        '-algorithm_label',
        action='store',
        dest='ALG_LABEL',
        default=None,
        type=str,
        help='Optional label used for output filenames and algorithm field.',
    )
    parser.add_argument('-results_path', action='store', dest='RDIR',
                        default='results_test', type=str, 
                        help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=42, type=int, help='Seed / trial')
    parser.add_argument('-test',action='store_true', dest='TEST', 
                       help='Used for testing a minimal version')
    parser.add_argument('-n_jobs',action='store',  type=str, default='4',
                        help='number of cores available')
    parser.add_argument('-max_samples',action='store',  type=int, default=0,
                        help='number of training samples')
    parser.add_argument('-target_noise',action='store',dest='Y_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-feature_noise',action='store',dest='X_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-fit_time_limit',action='store',dest='FITTIME',default=3600,
            type=int, help='Fit time limit (seconds) e.g. 3600 (1 hour). This is the maximum time for the fit method, not the job, make sure job time lim is greater than this.')
    parser.add_argument('--sym_data', action='store_true', dest='SYM_DATA', default=False)
    parser.add_argument('--save_population', action='store_true', dest='SAVE_POP', default=False)
    parser.add_argument('--ecotracker', action='store_true', dest='ECOTRACKER', default=False)
    parser.add_argument('--scale_x', action='store_true', dest='SCALE_X', default=False) 
    parser.add_argument('--scale_y', action='store_true', dest='SCALE_Y', default=False)
    parser.add_argument('--skip_tuning',action='store_true', dest='SKIP_TUNE', 
                        default=False, help='Dont tune the estimator')
    parser.add_argument('--tuned',action='store_true', dest='TUNED', default=False, 
            help='Run tuned version of estimators. Only applies when ml=None')
    
    args = parser.parse_args()
    set_env_vars(args.n_jobs)
    # import algorithm 
    print('import from','methods.'+args.ALG+'.regressor')
    algorithm = importlib.__import__('methods.'+args.ALG+'.regressor',
                                     globals(),
                                     locals(),
                                     ['*']
                                    )

    print('algorithm:',algorithm.est)

    # optional keyword arguments passed to evaluate
    eval_kwargs, test_params = {},{}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    if args.max_samples != 0:
        eval_kwargs['max_train_samples'] = args.max_samples

    eval_kwargs['sym_data'] = args.SYM_DATA
    eval_kwargs['save_pop'] = args.SAVE_POP
    eval_kwargs['scale_x'] = args.SCALE_X
    eval_kwargs['scale_y'] = args.SCALE_Y

    evaluate_model(
        dataset=args.INPUT_FILE,
        results_path=args.RDIR,
        random_state=args.RANDOM_STATE,
        est_name=(args.ALG_LABEL if args.ALG_LABEL is not None else args.ALG),
        est=algorithm.est,  
        model=algorithm.model,
        algorithm=algorithm,
        
        ecotracker=args.ECOTRACKER,
        test=args.TEST,
        target_noise=args.Y_NOISE, 
        feature_noise=args.X_NOISE, 
        use_tuned=args.TUNED,
        fit_time_limit=args.FITTIME,
        
        **eval_kwargs
    )
