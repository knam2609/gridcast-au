"""Property test: forecasts at an origin are invariant to any change in unknown data.

For random series, origins and latencies, every value after the information cutoff
(``position > origin - latency``) is replaced by random noise; every baseline's forecast
for that origin must be bit-for-bit identical.
"""

from functools import cache

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from gridcast.forecasting.origins import ForecastSetup, SeriesContext, origin_positions
from gridcast.models.baselines import default_baselines
from tests.forecast_fixtures import make_ctx


@cache
def _ctx() -> SeriesContext:
    rng = np.random.default_rng(42)
    return make_ctx(days=40, values=lambda pos: rng.normal(5000, 500, len(pos)))


@settings(max_examples=60, deadline=None)
@given(
    origin_rank=st.integers(min_value=0, max_value=10_000),
    latency=st.sampled_from([0, 1, 7]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_forecasts_do_not_depend_on_future_data(origin_rank: int, latency: int, seed: int) -> None:
    ctx = _ctx()
    setup = ForecastSetup(latency_steps=latency)
    candidates = origin_positions(ctx, setup, ctx.times[7 * 288 + 50])
    origin = candidates[origin_rank % len(candidates)]
    origins = np.array([origin])

    rng = np.random.default_rng(seed)
    perturbed = ctx.y.copy()
    unknown = np.arange(len(ctx)) > origin - latency
    perturbed[unknown] = rng.normal(-1e5, 1e5, unknown.sum())
    future_ctx = ctx.with_y(perturbed)

    for model in default_baselines("price"):  # superset of the demand baselines
        before = model.predict(ctx, origins, setup)
        after = model.predict(future_ctx, origins, setup)
        np.testing.assert_array_equal(before, after, err_msg=model.name)
