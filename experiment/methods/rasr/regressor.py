import hashlib
import itertools
import os
import re

import numpy as np
import sympy as sp
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.validation import check_array

from src.rasr.inference.rasr import RASR

MAX_FEATURES = 11
MAX_SUBSAMPLE_SIZE = 2048
MIN_VAL_SIZE = 256

_DEFAULT_NUM_CANDIDATES = int(os.environ.get("RASR_NUM_CANDIDATES", 1024))
_DEFAULT_TEMPERATURE = float(os.environ.get("RASR_TEMPERATURE", 1.0))
_DEFAULT_N_RESTARTS = int(os.environ.get("RASR_N_RESTARTS", 4))
_DEFAULT_INIT_NOISE_SCALE = float(os.environ.get("RASR_INIT_NOISE_SCALE", 0.5))
_DEFAULT_REFINE_CONSTANTS = os.environ.get("RASR_REFINE_CONSTANTS", "1") != "0"
_DEFAULT_REFINER_WORKERS = int(os.environ.get("REFINER_WORKERS", 0))


def _replace_variable_names(expr, columns):
    if columns is None:
        return expr

    pattern = re.compile(r"\bx(\d+)\b")
    return pattern.sub(
        lambda m: columns[int(m.group(1)) - 1]
        if 0 < int(m.group(1)) <= len(columns)
        else m.group(0),
        expr,
    )


def _resolve_model_dir(local_model_root, model_subdir):
    root = local_model_root if local_model_root else "/srbench_pretrained"
    root = os.path.expanduser(root)
    model_dir = os.path.join(root, model_subdir) if model_subdir else root
    return os.path.abspath(model_dir)

_BACKEND_CACHE = {}
_CANDIDATE_CACHE = {}


def _get_cached_backend(model_dir, device, dtype, regressor):
    """Return a RASR for model_dir, loading once.

    On a cache hit the already-loaded weights are reused and only the sampling
    hyperparameters (which fit() reads from the instance) are refreshed for the
    current grid config.
    """
    key = (model_dir, device, str(dtype))
    backend = _BACKEND_CACHE.get(key)
    if backend is None:
        backend = RASR(
            model_path=model_dir,
            device=device,
            dtype=dtype,
            verbose=True,
            num_candidates=regressor.num_candidates,
            temperature=regressor.temperature,
            top_p=regressor.top_p,
            n_restarts=regressor.n_restarts,
            decode_batch_size=regressor.decode_batch_size,
            scoring="r2",
            parsimony=regressor.parsimony,
            init_noise_scale=regressor.init_noise_scale,
            refine_constants=regressor.refine_constants,
            refiner_workers=regressor.refiner_workers,
        )
        _BACKEND_CACHE[key] = backend
        print(f"[RASR] loaded checkpoint into cache: {model_dir}", flush=True)
    else:
        # Reuse the already-loaded weights.
        backend.num_candidates = int(regressor.num_candidates)
        backend.temperature = float(regressor.temperature)
        backend.top_p = float(regressor.top_p)
        backend.n_restarts = int(regressor.n_restarts)
        backend.decode_batch_size = int(regressor.decode_batch_size)
        backend.parsimony = float(regressor.parsimony)
        backend.init_noise_scale = float(regressor.init_noise_scale)
        backend.refine_constants = bool(regressor.refine_constants)
        print(f"[RASR] reusing cached checkpoint: {model_dir}", flush=True)
    return backend


def _data_fingerprint(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def _select_by_parsimony(est, parsimony):
    """
    Re-pick the best candidate from an already-generated/refined pool for a
    given parsimony weight, without regenerating or re-refining anything.

    Prefers val_r2 over r2 when the pool was fit with a val split (val_r2 finite),
    matching the scoring RASR.fit() itself applies -- so caching doesn't change which
    candidate wins relative to an uncached call with the same parsimony.
    """
    def _score(r):
        effective_r2 = r.val_r2 if np.isfinite(r.val_r2) else r.r2
        return effective_r2 - parsimony * r.complexity

    best_idx = max(range(len(est.results_)), key=lambda i: _score(est.results_[i]))
    est.results_[0], est.results_[best_idx] = est.results_[best_idx], est.results_[0]
    est.best_result_ = est.results_[0]


def reset_backend_cache():
    _BACKEND_CACHE.clear()
    _CANDIDATE_CACHE.clear()


class RASRRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        random_state=42,
        local_model_root="/srbench_pretrained",
        model_subdir="",
        num_candidates=_DEFAULT_NUM_CANDIDATES,
        temperature=_DEFAULT_TEMPERATURE,
        top_p=0.95,
        n_restarts=_DEFAULT_N_RESTARTS,
        init_noise_scale: float = _DEFAULT_INIT_NOISE_SCALE,
        refine_constants: bool = _DEFAULT_REFINE_CONSTANTS,
        refiner_workers: int = _DEFAULT_REFINER_WORKERS,
        decode_batch_size=32,
        simplify_expressions=False,
        variable_names_mode="auto",
        parsimony=0.0,
    ):
        self.random_state = random_state
        self.local_model_root = local_model_root
        self.model_subdir = model_subdir
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.top_p = top_p
        self.n_restarts = n_restarts
        self.init_noise_scale = init_noise_scale
        self.refine_constants = refine_constants
        self.refiner_workers = refiner_workers
        self.decode_batch_size = decode_batch_size
        self.simplify_expressions = simplify_expressions
        self.variable_names_mode = variable_names_mode
        self.parsimony = parsimony
        self.variance_threshold = 0.01

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

        self.model_dir_ = _resolve_model_dir(self.local_model_root, self.model_subdir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device.startswith("cuda") else None
        self.device_ = device
        if torch.cuda.is_available():
            print(f"[RASR] GPU: {torch.cuda.get_device_name(0)}, dtype: bfloat16", flush=True)

        self.est_ = _get_cached_backend(self.model_dir_, device, dtype, self)

        X_fit = check_array(X)
        y_fit = np.asarray(y).reshape(-1)

        X_val = None
        y_val = None
        if X_fit.shape[0] > MAX_SUBSAMPLE_SIZE:
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.choice(X_fit.shape[0], size=MAX_SUBSAMPLE_SIZE, replace=False)
            remaining_idx = np.setdiff1d(np.arange(X_fit.shape[0]), sample_idx, assume_unique=True)
            if remaining_idx.shape[0] >= MIN_VAL_SIZE:
                X_val = X_fit[remaining_idx]
                y_val = y_fit[remaining_idx]
            X_fit = X_fit[sample_idx]
            y_fit = y_fit[sample_idx]
        self.selected_feature_indices_ = np.arange(X_fit.shape[1], dtype=int)

        if X_fit.shape[1] > MAX_FEATURES:
            variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            X_var = variance_selector.fit_transform(X_fit)
            if X_var.shape[1] > 0:
                var_kept_indices = np.where(variance_selector.get_support())[0]
            else:
                # Fallback: keep all features if thresholding removes everything.
                X_var = X_fit
                var_kept_indices = np.arange(X_fit.shape[1], dtype=int)

            if X_var.shape[1] > MAX_FEATURES:
                y_centered = y_fit - np.mean(y_fit)
                x_centered = X_var - np.mean(X_var, axis=0)
                denom = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))

                correlations = np.zeros(X_var.shape[1], dtype=float)
                valid = denom > 0
                if np.any(valid):
                    correlations[valid] = np.abs(
                        np.sum(x_centered[:, valid] * y_centered[:, None], axis=0)
                        / denom[valid]
                    )

                top_local_indices = np.argsort(correlations)[-MAX_FEATURES:]
                top_local_indices = top_local_indices[
                    np.argsort(correlations[top_local_indices])[::-1]
                ]
                self.selected_feature_indices_ = var_kept_indices[top_local_indices]
            else:
                self.selected_feature_indices_ = var_kept_indices

        X_fit = X_fit[:, self.selected_feature_indices_]
        if X_val is not None:
            X_val = X_val[:, self.selected_feature_indices_]

        cache_key = (self.model_dir_, _data_fingerprint(X_fit, y_fit))
        cached_results = _CANDIDATE_CACHE.get(cache_key)

        if cached_results is not None:
            print("[RASR] reusing cached candidate pool for this training data "
                  "(skipping generation+refinement, only parsimony differs)", flush=True)
            self.est_.results_ = list(cached_results)
        else:
            self.est_.fit(
                X_fit,
                y_fit,
                variable_names=self.variable_names_mode,
                simplify_expressions=self.simplify_expressions,
                X_val=X_val,
                y_val=y_val,
            )
            _CANDIDATE_CACHE[cache_key] = list(self.est_.results_)

        _select_by_parsimony(self.est_, self.parsimony)

        # High precision necessary, otherwise parsing can fail on constants that round to 0.
        self.expression_ = str(self.est_.get_expression(precision=8))
        return self

    def oracle_select(self, X_test, y_test):
        from src.rasr.inference.metrics import r2_score as _r2
        X = check_array(X_test)
        if X.shape[1] != len(self.selected_feature_indices_):
            if X.shape[1] > np.max(self.selected_feature_indices_):
                X = X[:, self.selected_feature_indices_]
        y = np.asarray(y_test).reshape(-1)
        best_r2, best_idx = -np.inf, 0
        for i in range(len(self.est_.results_)):
            y_pred = self.est_.predict(X, nth_best_beam=i).reshape(-1)
            r2 = float(_r2(y, y_pred))
            if np.isfinite(r2) and r2 > best_r2:
                best_r2, best_idx = r2, i
        self.est_.results_[0], self.est_.results_[best_idx] = (
            self.est_.results_[best_idx],
            self.est_.results_[0],
        )
        self.est_.best_result_ = self.est_.results_[0]
        # High precision necessary, otherwise parsing can fail on constants that round to 0.
        self.expression_ = str(self.est_.get_expression(precision=8))

    def predict(self, X):
        X = check_array(X)
        if X.shape[1] != len(self.selected_feature_indices_):
            if X.shape[1] > np.max(self.selected_feature_indices_):
                X = X[:, self.selected_feature_indices_]
            else:
                raise ValueError(
                    "Input feature count does not match the trained feature subset."
                )
        return np.asarray(self.est_.predict(X))


est = RASRRegressor()


def model(est, X=None):
    expr = str(est.expression_).replace("^", "**")
    if X is not None and hasattr(X, "columns"):
        columns = list(X.columns)
        if hasattr(est, "selected_feature_indices_"):
            if len(columns) > np.max(est.selected_feature_indices_):
                columns = [columns[idx] for idx in est.selected_feature_indices_]
        expr = _replace_variable_names(expr, columns)
    return expr


def complexity(est):
    try:
        return int(sp.count_ops(sp.sympify(model(est))))
    except Exception:
        return -1


eval_kwargs = dict(
    test_params={"num_candidates": 8, "n_restarts": 1, "decode_batch_size": 16},
    use_dataframe=False,
)

_PARSIMONY_VALUES = [0.0, 0.01, 0.1]

hyper_params = [
    {
        'parsimony':      (p,),
    }
    for p in _PARSIMONY_VALUES
]
