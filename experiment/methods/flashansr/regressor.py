import hashlib
import inspect
import os
import re

import numpy as np
import sympy as sp
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.validation import check_array

from flash_ansr import FlashANSR, SoftmaxSamplingConfig


MAX_FEATURES = 17

_DEFAULT_REFINER_WORKERS = int(os.environ.get("REFINER_WORKERS", 0))

_BACKEND_CACHE = {}
_CANDIDATE_CACHE = {}


def _get_cached_backend(model_dir, device, regressor):
    """Return a FlashANSR for model_dir, loading once.

    On a cache hit the already-loaded weights are reused and only the sampling
    hyperparameters (which fit() reads from the instance) are refreshed for the
    current grid config.
    """
    key = (model_dir, str(device))
    backend = _BACKEND_CACHE.get(key)
    if backend is None:
        backend = FlashANSR.load(
            directory=model_dir,
            generation_config=SoftmaxSamplingConfig(choices=regressor.choices),
            n_restarts=regressor.n_restarts,
            parsimony=regressor.parsimony,
            device=device,
            refiner_workers=regressor.refiner_workers,
        )
        _BACKEND_CACHE[key] = backend
        print(f"[FlashANSR] loaded checkpoint into cache: {model_dir}", flush=True)
    else:
        # Reuse the already-loaded weights.
        backend.generation_config = SoftmaxSamplingConfig(choices=regressor.choices)
        backend.n_restarts = int(regressor.n_restarts)
        backend.parsimony = float(regressor.parsimony)
        print(f"[FlashANSR] reusing cached checkpoint: {model_dir}", flush=True)
    return backend


def _data_fingerprint(X, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def reset_backend_cache():
    _BACKEND_CACHE.clear()
    _CANDIDATE_CACHE.clear()


def _replace_variable_names(expr, columns):
    if columns is None:
        return expr

    # FlashANSR v0.4.5 emits x1, x2, ...
    # Keep x_1 as a narrow compatibility fallback.
    pattern = re.compile(r"\bx_?(\d+)\b")
    expr = pattern.sub(
        lambda m: columns[int(m.group(1)) - 1]
        if 0 < int(m.group(1)) <= len(columns)
        else m.group(0),
        expr,
    )

    return expr


class FlashANSRRegressor(BaseEstimator, RegressorMixin):
    # TODO: Return constants in higher precision to avoid mistakes in SymPy parsing.
    def __init__(
        self,
        random_state=42,
        model_dir="/srbench_pretrained",
        choices=16384,  # Largest in their "short" range.
        n_restarts=8,
        parsimony=0.0,
        refiner_workers=_DEFAULT_REFINER_WORKERS,
    ):
        self.random_state = random_state
        # Directory holding the pretrained checkpoint files directly:
        # model.yaml, tokenizer.yaml, state_dict.pt.
        self.model_dir = model_dir
        self.choices = choices
        self.n_restarts = n_restarts
        self.parsimony = parsimony
        self.refiner_workers = refiner_workers
        self.variance_threshold = 0.01

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

        required_files = ["model.yaml", "tokenizer.yaml", "state_dict.pt"]
        missing = [
            fname
            for fname in required_files
            if not os.path.isfile(os.path.join(self.model_dir, fname))
        ]
        if missing:
            raise FileNotFoundError(
                f"FlashANSR checkpoint incomplete in {self.model_dir!r}; "
                f"missing: {', '.join(missing)}"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"FlashANSR using device: {device}", flush=True)
        self.device_ = device

        self.est_ = _get_cached_backend(self.model_dir, device, self)

        X = check_array(X)
        y = np.asarray(y).reshape(-1)

        if X.shape[0] > 2048:
            rng = np.random.RandomState(self.random_state)
            sample_idx = rng.choice(X.shape[0], size=2048, replace=False)
            X = X[sample_idx]
            y = y[sample_idx]

        self.selected_feature_indices_ = np.arange(X.shape[1], dtype=int)

        if X.shape[1] > MAX_FEATURES:
            variance_selector = VarianceThreshold(threshold=self.variance_threshold)
            X_var = variance_selector.fit_transform(X)
            if X_var.shape[1] > 0:
                var_kept_indices = np.where(variance_selector.get_support())[0]
            else:
                # Fallback: keep all features if thresholding removes everything.
                X_var = X
                var_kept_indices = np.arange(X.shape[1], dtype=int)

            if X_var.shape[1] > MAX_FEATURES:
                y_centered = y - np.mean(y)
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

        X_fit = X[:, self.selected_feature_indices_]

        cache_key = (self.model_dir, _data_fingerprint(X_fit, y))
        cached_results = _CANDIDATE_CACHE.get(cache_key)

        if cached_results is not None:
            print("[FlashANSR] reusing cached candidate pool for this training data "
                  "(skipping generation+refinement, only parsimony differs)", flush=True)
            self.est_._results = list(cached_results)
            self.est_.compile_results(self.parsimony)
        else:
            self.est_.fit(X_fit, y, verbose=True)
            _CANDIDATE_CACHE[cache_key] = list(self.est_._results)

        if hasattr(self.est_, "get_expression"):
            self.expression_ = str(self.est_.get_expression())
        else:
            self.expression_ = str(self.est_.results.iloc[0]["expression"])

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
        return np.asarray(self.est_.predict(X))


est = FlashANSRRegressor()


def model(est, X=None):
    expr = str(est.expression_)
    expr = expr.replace("^", "**")
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
    test_params={"choices": 8, "n_restarts": 1},
    use_dataframe=False,
)

hyper_params = [
    {'parsimony': (0.0,)},
    {'parsimony': (0.05,)},
    {'parsimony': (0.1,)},
]


