import os
import re
import random
from collections import defaultdict

import numpy as np
import sympy as sp
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array

# PyTorch >=2.6 defaults torch.load(..., weights_only=True), which refuses to
# unpickle non-tensor objects (e.g. the pathlib.PosixPath baked into
# snip-e2e-sr.pth's saved training config) unless explicitly allow-listed.
# symbolicregression/model/__init__.py::build_modules() calls
# `torch.load(params.reload_model)` with no weights_only argument -- that's
# vendored code we don't want to patch in site-packages, so restore the old
# default here instead. Safe since the checkpoint is from a trusted source
# (SNIP's own release).
_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat

import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.model.sklearn_wrapper import get_top_k_features
import symbolicregression.model.utils_wrapper as utils_wrapper
from parsers import get_parser
from model import SNIPSymbolicRegressor
from LSO_fit import lso_fit, LSOFitNeverGrad

MAX_SUBSAMPLE_SIZE = 2048


def _replace_variable_names(expr, columns):
    # SNIP's tree.infix() emits 0-indexed variables x_0, x_1, ...
    if columns is None:
        return expr
    pattern = re.compile(r"\bx_(\d+)\b")
    return pattern.sub(
        lambda m: columns[int(m.group(1))]
        if 0 <= int(m.group(1)) < len(columns)
        else m.group(0),
        expr,
    )


def _resolve_checkpoint(local_model_root, checkpoint_name):
    root = local_model_root if local_model_root else "/srbench_pretrained"
    root = os.path.expanduser(root)
    return os.path.abspath(os.path.join(root, checkpoint_name))


def _build_params(
    device,
    checkpoint_path,
    lso_optimizer,
    lso_pop_size,
    lso_max_iteration,
    lso_stop_r2,
    beam_size,
    max_input_points,
    seed,
):
    # Mirrors the overrides applied in snip/LSO_eval.py's __main__ block --
    # these are load-bearing defaults matching how the released checkpoint
    # (snip-e2e-sr.pth) was trained (10-d encoder, 16 decoder layers, etc.).
    params = get_parser().parse_args([])
    params.reload_model = checkpoint_path
    params.device = device
    params.cpu = device != "cuda"
    params.batch_size = 1
    params.batch_size_eval = int(1.5 * params.batch_size)
    params.n_steps_per_epoch = 100
    params.max_input_dimension = 10
    params.env_base_seed = 2023
    params.n_dec_layers = 16
    params.local_rank = -1
    params.master_port = -1
    params.num_workers = 1
    params.max_number_bags = 1
    params.rescale = True
    params.eval_only = True
    params.seed = seed

    params.lso_optimizer = lso_optimizer
    params.lso_pop_size = lso_pop_size
    params.lso_max_iteration = lso_max_iteration
    params.lso_stop_r2 = lso_stop_r2
    params.beam_size = beam_size
    params.max_input_points = max_input_points
    params.n_trees_to_refine = beam_size
    return params


# Building the env/model from a checkpoint is expensive (loads the SNIP
# encoder + E2E decoder). HPO only varies lightweight LSO search params
# (lso_optimizer/pop_size/iterations/beam_size), so cache the heavy part
# across grid points / CV folds, keyed by the checkpoint actually loaded.
# GridSearchCV runs single-threaded here, so a plain dict is safe.
_BACKEND_CACHE = {}


def _get_cached_backend(checkpoint_path, device, params):
    key = (checkpoint_path, device)
    cached = _BACKEND_CACHE.get(key)
    if cached is None:
        symbolicregression.utils.CUDA = device == "cuda"
        env = build_env(params)
        env.rng = np.random.RandomState(0)
        # build_modules() reloads params.reload_model into each module's
        # state_dict internally when params.reload_model != "" (see
        # symbolicregression/model/__init__.py), so no separate reload step
        # is needed here.
        modules = build_modules(env, params)
        model = SNIPSymbolicRegressor(params=params, env=env, modules=modules)
        model.to(device)
        cached = (env, model)
        _BACKEND_CACHE[key] = cached
        print(f"[SNIP] loaded checkpoint into cache: {checkpoint_path}", flush=True)
    else:
        print(f"[SNIP] reusing cached checkpoint: {checkpoint_path}", flush=True)
    env, model = cached
    # Refresh the LSO/beam search hyperparameters this call's grid point
    # wants -- model.generate_from_latent_sampling() reads them off
    # model.params directly.
    model.params = params
    return env, model


def reset_backend_cache():
    _BACKEND_CACHE.clear()


class SNIPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        random_state=42,
        local_model_root="/srbench_pretrained",
        checkpoint_name="snip-e2e-sr.pth",
        lso_optimizer="gwo",
        lso_pop_size=15,
        lso_max_iteration=10,
        lso_stop_r2=0.99,
        beam_size=2,
        max_input_points=200,
    ):
        self.random_state = random_state
        self.local_model_root = local_model_root
        self.checkpoint_name = checkpoint_name
        self.lso_optimizer = lso_optimizer
        self.lso_pop_size = lso_pop_size
        self.lso_max_iteration = lso_max_iteration
        self.lso_stop_r2 = lso_stop_r2
        self.beam_size = beam_size
        self.max_input_points = max_input_points

    def fit(self, X, y):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if not torch.cuda.is_available():
            # snip/LSO_fit.py has several unconditional `.cuda()` calls
            # (population update, beam decoding); there is no CPU fallback.
            raise RuntimeError(
                "SNIP requires a CUDA GPU (upstream snip/LSO_fit.py has "
                "hardcoded .cuda() calls with no CPU path)."
            )
        torch.cuda.manual_seed(self.random_state)
        device = "cuda"

        checkpoint_path = _resolve_checkpoint(self.local_model_root, self.checkpoint_name)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"SNIP checkpoint not found at {checkpoint_path!r}. Download "
                "snip-e2e-sr.pth (see snip/README.md) and place it under "
                "pretrained/SNIP/."
            )

        params = _build_params(
            device=device,
            checkpoint_path=checkpoint_path,
            lso_optimizer=self.lso_optimizer,
            lso_pop_size=self.lso_pop_size,
            lso_max_iteration=self.lso_max_iteration,
            lso_stop_r2=self.lso_stop_r2,
            beam_size=self.beam_size,
            max_input_points=self.max_input_points,
            seed=self.random_state,
        )
        self.env_, self.model_ = _get_cached_backend(checkpoint_path, device, params)
        self.params_ = params

        X = check_array(X)
        y = np.asarray(y).reshape(-1)

        # The released checkpoint was trained with max_input_dimension=10.
        self.selected_feature_indices_ = np.arange(X.shape[1], dtype=int)
        if X.shape[1] > params.max_input_dimension:
            top_k = get_top_k_features(X, y, k=params.max_input_dimension)
            self.selected_feature_indices_ = np.array(top_k, dtype=int)
        X_fit = X[:, self.selected_feature_indices_]
        y_fit = y

        if X_fit.shape[0] > MAX_SUBSAMPLE_SIZE:
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.choice(X_fit.shape[0], size=MAX_SUBSAMPLE_SIZE, replace=False)
            X_fit = X_fit[sample_idx]
            y_fit = y_fit[sample_idx]

        y_fit = y_fit.reshape(-1, 1)

        # Only the encoder's input is additionally standardized here (SNIP's
        # own convention -- see snip/LSO_eval.py::evaluate_pmlb_lso); the
        # decoded tree's constants are fit/refined against the raw
        # (already SRBench-scaled) X_fit/y_fit below.
        scaler = utils_wrapper.StandardScaler()
        X_scaled = scaler.fit_transform(X_fit)

        # SRBench already provides a train/test split; SNIP's own LSO code
        # additionally expects a "to_predict" slot it doesn't actually need
        # for our purposes (SRBench calls est.predict() separately), so we
        # just reuse the fit slice there.
        sample_to_learn = {
            "X_scaled_to_fit": [X_scaled],
            "Y_scaled_to_fit": [y_fit],
            "x_to_fit": [X_fit],
            "y_to_fit": [y_fit],
            "x_to_predict": [X_fit],
            "y_to_predict": [y_fit],
        }

        batch_results = defaultdict(list)
        with torch.no_grad():
            if params.lso_optimizer == "gwo":
                batch_results = lso_fit(
                    sample_to_learn, self.env_, params, self.model_, batch_results, bag_number=1
                )
            else:
                opt = LSOFitNeverGrad(
                    self.env_, params, self.model_, sample_to_learn, batch_results, bag_number=1
                )
                batch_results = opt.fit_func()

        best_eq = batch_results.get("final_predicted_tree", [None])[0]
        if best_eq is None or isinstance(best_eq, str):
            raise RuntimeError("SNIP failed to find a valid expression for this dataset.")
        self.tree_ = best_eq
        return self

    def predict(self, X):
        X = check_array(X)
        if X.shape[1] != len(self.selected_feature_indices_):
            if X.shape[1] > np.max(self.selected_feature_indices_):
                X = X[:, self.selected_feature_indices_]
            else:
                raise ValueError(
                    "Input feature count does not match the trained feature subset."
                )
        numexpr_fn = self.env_.simplifier.tree_to_numexpr_fn(self.tree_)
        return np.asarray(numexpr_fn(X)[:, 0]).reshape(-1)


est = SNIPRegressor()


def model(est, X=None):
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    expr = est.tree_.infix()
    for op, replace_op in replace_ops.items():
        expr = expr.replace(op, replace_op)

    if X is not None and hasattr(X, "columns"):
        columns = list(X.columns)
        if hasattr(est, "selected_feature_indices_"):
            if len(columns) > np.max(est.selected_feature_indices_, initial=0):
                columns = [columns[idx] for idx in est.selected_feature_indices_]
        expr = _replace_variable_names(expr, columns)
    return expr


def complexity(est):
    try:
        return int(sp.count_ops(sp.sympify(model(est))))
    except Exception:
        try:
            return len(est.tree_.prefix().split(","))
        except Exception:
            return -1


eval_kwargs = dict(
    test_params={"lso_pop_size": 5, "lso_max_iteration": 3, "beam_size": 2},
    use_dataframe=False,
)

# SNIP's own SRBench evaluation (snip/run_lso.sh) does not grid-search these
# hyperparameters -- it runs one fixed configuration. Reproducing that means
# skipping SRBench's optional tuning step entirely (run with --skip_tuning)
# rather than inventing a grid; each grid point here would cost a full
# 3-fold CV refit of an already expensive LSO search for no basis in the
# paper. The __init__ defaults above are a deliberately smaller/faster LSO
# budget than the SNIP paper's lso_pop_size=50/lso_max_iteration=80, to keep
# the runtime manageable.
hyper_params = []
