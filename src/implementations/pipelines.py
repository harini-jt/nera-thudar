from functools import partial

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

from src.implementations.improvements import (
    QuantWithDepthDiv,
    QuantWithIntervalStats,
    QuantWithSamplingFrequency,
    RandomFeatureDropout,
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


def build_random_dropout(
    seed=42,
    depth=6,
    div=4,
    n_estimators=200,
    max_features=0.1,
    criterion="entropy",
    n_jobs=-1,
    dropout_rate=0.2,
):
    """Quant, then drop a random fraction of feature columns, then ExtraTrees."""
    return make_pipeline(
        Quant(depth=depth, div=div),
        RandomFeatureDropout(rate=dropout_rate, seed=seed),
        _forest(seed, n_estimators, max_features, criterion, n_jobs),
    )


def build_sampling_frequency_dropout(
    seed=42,
    depth=6,
    div=4,
    n_estimators=200,
    max_features=0.1,
    criterion="entropy",
    n_jobs=-1,
    depth_repeats=None,
    dropout_rate=0.2,
):
    """Duplicate large-interval features, then randomly drop feature columns."""
    return make_pipeline(
        QuantWithSamplingFrequency(depth=depth, div=div, depth_repeats=depth_repeats),
        RandomFeatureDropout(rate=dropout_rate, seed=seed),
        _forest(seed, n_estimators, max_features, criterion, n_jobs),
    )


def build_interval_stats(
    seed=42,
    depth=6,
    div=4,
    n_estimators=200,
    max_features=0.1,
    criterion="entropy",
    n_jobs=-1,
):
    """Quant with std and slope added for every interval."""
    return make_pipeline(
        QuantWithIntervalStats(depth=depth, div=div),
        _forest(seed, n_estimators, max_features, criterion, n_jobs),
    )


# Method name (used in the CSV and on the command line) -> pipeline builder.
PIPELINES = {
    "quant_base": build_quant,
    "quant_gini": partial(build_quant, criterion="gini"),
    "dup_d1-2": partial(build_sampling_frequency, depth_repeats={1: 2, 2: 2}),
    "dup_d1-3": partial(build_sampling_frequency, depth_repeats={1: 2, 2: 2, 3: 2}),
    "dup_d1_4_d2_2": partial(build_sampling_frequency, depth_repeats={1: 4, 2: 2}),
    "random_dropout_0.1": partial(build_random_dropout, dropout_rate=0.1),
    "random_dropout_0.2": partial(build_random_dropout, dropout_rate=0.2),
    "random_dropout_0.5": partial(build_random_dropout, dropout_rate=0.5),
    "dup_d1-2_dropout_0.1": partial(
        build_sampling_frequency_dropout, depth_repeats={1: 2, 2: 2}, dropout_rate=0.1
    ),
    "dup_d1-2_dropout_0.2": partial(
        build_sampling_frequency_dropout, depth_repeats={1: 2, 2: 2}, dropout_rate=0.2
    ),
    "dup_d1_4_d2_2_dropout_0.1": partial(
        build_sampling_frequency_dropout, depth_repeats={1: 4, 2: 2}, dropout_rate=0.1
    ),
    "dup_d1_4_d2_2_dropout_0.2": partial(
        build_sampling_frequency_dropout, depth_repeats={1: 4, 2: 2}, dropout_rate=0.2
    ),
    "quant_stats": build_interval_stats,
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
