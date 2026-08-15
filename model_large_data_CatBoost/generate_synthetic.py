"""
Generate a large synthetic dataset by sampling the 5 operating-condition
inputs across (and slightly beyond) the ranges observed in train/test, then
running each sample through the mechanistic PFR ODE model fit on the real
150-row train_dataset.csv (see mechanistic_pfr.py / mechanistic_params.npy).

This gives CatBoost a much larger, physically-consistent training set to
learn the non-linear yield surface (esp. the sharp zero-yield cliff) without
needing external plant data, which doesn't exist for this specific reactor.

We deliberately sample a bit beyond the observed min/max (5% margin) so the
model sees the edges of the operating envelope, but not wildly out of
distribution.
"""
import numpy as np
import pandas as pd
from mechanistic_pfr import predict_batch

RNG = np.random.default_rng(42)

FEATURES = [
    "flow_rate_L_min", "concentration_mol_L", "inlet_temperature_K",
    "length_m", "jacket_temperature_K",
]


def get_ranges(train_csv, test_csv, margin=0.05):
    df = pd.concat([pd.read_csv(train_csv)[FEATURES], pd.read_csv(test_csv)[FEATURES]])
    ranges = {}
    for c in FEATURES:
        lo, hi = df[c].min(), df[c].max()
        span = hi - lo
        ranges[c] = (lo - margin * span, hi + margin * span)
    return ranges


def sample_uniform(ranges, n):
    data = {c: RNG.uniform(lo, hi, n) for c, (lo, hi) in ranges.items()}
    return pd.DataFrame(data)


def sample_lhs(ranges, n):
    """Latin Hypercube sampling for better space coverage than plain uniform."""
    d = len(ranges)
    cut = np.linspace(0, 1, n + 1)
    u = RNG.uniform(size=(n, d))
    a = cut[:n]
    b = cut[1:n + 1]
    pts = np.zeros((n, d))
    for j in range(d):
        perm = RNG.permutation(n)
        pts[:, j] = a[perm] + u[:, j] * (b[perm] - a[perm])

    data = {}
    for j, (c, (lo, hi)) in enumerate(ranges.items()):
        data[c] = lo + pts[:, j] * (hi - lo)
    return pd.DataFrame(data)


def generate(train_csv, test_csv, n_samples=5000, out_csv="synthetic_dataset.csv"):
    ranges = get_ranges(train_csv, test_csv)
    df = sample_lhs(ranges, n_samples)

    params = np.load("mechanistic_params.npy")
    yields = predict_batch(df, params)

    df["overall_yield"] = np.round(yields, 3)
    df.to_csv(out_csv, index=False)
    print(f"Generated {n_samples} synthetic rows -> {out_csv}")
    print(f"  zero-yield fraction: {(df.overall_yield == 0).mean():.2%}")
    print(df.describe())
    return df


if __name__ == "__main__":
    generate("train_dataset.csv", "test_dataset.csv", n_samples=5000)
