import numpy as np
import pytest

from gridcast.evaluation.metrics import masked_metrics


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    actual = rng.normal(5000, 500, (200, 12))
    forecast = actual + rng.normal(10, 100, actual.shape)
    mask = rng.random(actual.shape) > 0.2
    return forecast, actual, mask


def test_pooled_metrics_match_numpy(data) -> None:  # type: ignore[no-untyped-def]
    f, a, m = data
    out, n = masked_metrics(f, a, m, ("mae", "rmse", "bias", "mape", "medae"))
    e = (f - a)[m]
    assert n == m.sum()
    assert out["mae"] == pytest.approx(np.mean(np.abs(e)))
    assert out["rmse"] == pytest.approx(np.sqrt(np.mean(e**2)))
    assert out["bias"] == pytest.approx(np.mean(e))
    assert out["mape"] == pytest.approx(100 * np.mean(np.abs(e) / np.abs(a[m])))
    assert out["medae"] == pytest.approx(np.median(np.abs(e)))


def test_per_column_metrics_match_numpy(data) -> None:  # type: ignore[no-untyped-def]
    f, a, m = data
    out, n = masked_metrics(f, a, m, ("mae", "medae"), axis=0)
    for j in range(a.shape[1]):
        e = (f[:, j] - a[:, j])[m[:, j]]
        assert n[j] == m[:, j].sum()
        assert out["mae"][j] == pytest.approx(np.mean(np.abs(e)))
        assert out["medae"][j] == pytest.approx(np.median(np.abs(e)))


def test_empty_mask_gives_nan(data) -> None:  # type: ignore[no-untyped-def]
    f, a, _ = data
    out, n = masked_metrics(f, a, np.zeros_like(a, dtype=bool), ("mae", "medae"))
    assert n == 0
    assert np.isnan(out["mae"])
    assert np.isnan(out["medae"])


def test_positive_bias_means_over_forecast() -> None:
    a = np.array([[100.0, 100.0]])
    out, _ = masked_metrics(a + 5, a, np.ones_like(a, dtype=bool), ("bias",))
    assert out["bias"] == pytest.approx(5.0)
