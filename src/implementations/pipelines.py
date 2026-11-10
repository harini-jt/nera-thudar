from functools import partial

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

from src.implementations.improvements import (
    QuantWithDepthDiv,
    QuantWithSamplingFrequency,
)
from src.implementations.quant import Quant


def _forest(seed, n_estimators, max_features, criterion, n_jobs):
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        criterion=criterion,
        random_state=seed,
        n_jobs=n_jobs,
    )


def build_quant(
    seed=42,
    depth=6,
    div=4,
    n_estimators=200,
    max_features=0.1,
    criterion="entropy",
    n_jobs=-1,
):
    """Quant + ExtraTreesClassifier with the paper's defaults."""
    return make_pipeline(
        Quant(depth=depth, div=div),
        _forest(seed, n_estimators, max_features, criterion, n_jobs),
    )


def build_sampling_frequency(
    seed=42,
    depth=6,
    div=4,
    n_estimators=200,
    max_features=0.1,
    criterion="entropy",
    n_jobs=-1,
    depth_repeats=None,
):
    return make_pipeline(
        QuantWithSamplingFrequency(depth=depth, div=div, depth_repeats=depth_repeats),
        _forest(seed, n_estimators, max_features, criterion, n_jobs),
    )


def build_depth_div(
    seed=42,
    depth=6,
    div=4,
    n_estimators=200,
    max_features=0.1,
    criterion="entropy",
    n_jobs=-1,
    depth_div=None,
):
    return make_pipeline(
        QuantWithDepthDiv(depth=depth, div=div, depth_div=depth_div),
        _forest(seed, n_estimators, max_features, criterion, n_jobs),
    )


# Method name (used in the CSV and on the command line) -> pipeline builder.
PIPELINES = {
    "quant_base": build_quant,
    "quant_gini": partial(build_quant, criterion="gini"),
    "dup_d1-2": partial(build_sampling_frequency, depth_repeats={1: 2, 2: 2}),
    "dup_d1-3": partial(build_sampling_frequency, depth_repeats={1: 2, 2: 2, 3: 2}),
    "depth_div_m2_m8": partial(
        build_depth_div, depth_div={1: 2, 2: 2, 3: 2, 5: 8, 6: 8}
    ),
}


def get_pipeline(method: str, **kwargs):
    if method not in PIPELINES:
        raise KeyError(f"Unknown method '{method}'. Available: {sorted(PIPELINES)}")
    return PIPELINES[method](**kwargs)


def method_config(method: str) -> dict:
    """The fixed keyword arguments a registered method was built with (for logging)."""
    return dict(getattr(PIPELINES[method], "keywords", {}))
