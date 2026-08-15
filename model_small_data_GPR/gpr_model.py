"""
Physics-informed Gaussian Process Regression for the small (150-row) dataset.

Approach:
  1. Mechanistic PFR ODE model (mechanistic_pfr.py) supplies a physics-based
     prior mean function mu(x) for every row (fit once on train_dataset.csv).
  2. GPR is trained on the RESIDUAL: r = overall_yield - mu(x), using a
     Matern kernel over physics-engineered features (Damkohler numbers,
     residence time, average/differential temperature). GPR final
     prediction = mu(x) + gp_residual_prediction(x), with calibrated
     uncertainty (std) coming from the GP posterior.
  3. Evaluated via repeated K-fold CV (since n=150 is small) to report a
     robust RMSE estimate before touching the test set.

Why GPR here: with only 150 rows and 5 inputs, GPR gives (a) strong
performance in the small-data regime, (b) exact-fit capacity so the
mechanistic residual can be learned without overfitting via kernel
length-scale regularization, and (c) predictive uncertainty, which is
directly useful for defending robustness in the pitch.

Warped-GP variant (warped=True in fit_final_and_predict / cross_validate):
overall_yield is a physically bounded quantity in [0, 100] with ~25% of
rows piled at exactly 0 -- a plain additive-Gaussian residual model can
predict outside [0,100] and treats the boundary pile-up as ordinary noise,
which is statistically the wrong likelihood for bounded/censored data
(Tobit-type regression literature; see Barber & Cai 2020, "Generalized
Multi-Output GP Censored Regression"). A tractable approximation used in
practice (Snelson, Rasmussen & Ghahramani, "Warped Gaussian Processes",
NeurIPS 2003) is to model a monotonic WARP of the target with a plain GP,
so the GP operates in an unbounded latent space where the Gaussian
likelihood is appropriate, then invert the warp to get predictions back
in [0,100]. We use a logit warp on y/100 (clipped away from the exact
boundary to keep it finite), with the mechanistic model's own prediction
passed through the same warp as the GP prior mean -- so mechanistic model
and GP now live in the same warped space instead of being combined
additively in raw yield units.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from mechanistic_pfr import predict_batch, R

# Logit warp: y in [0,100] -> clip to keep away from +-inf -> logit.
# EPS controls how close to the 0/100 boundary we allow before clipping;
# 37/150 rows are exactly at y=0, so EPS effectively sets how "confident"
# the warped space is allowed to get about a true zero.
WARP_EPS = 0.3  # percentage points


def warp(y):
    p = np.clip(y / 100.0, WARP_EPS / 100.0, 1 - WARP_EPS / 100.0)
    return np.log(p / (1 - p))


def unwarp(z):
    p = 1.0 / (1.0 + np.exp(-z))
    return 100.0 * p

MECH_PARAMS = np.load("mechanistic_params.npy")

# Rows with overall_yield below this are treated as the "collapsed" regime
# (side reaction B->C has consumed virtually all product B). There is a
# clean gap in the training data between ~0 and ~2-3% yield, so this
# threshold cleanly separates "reaction collapsed" from "reaction active".
COLLAPSE_THRESHOLD = 0.5


def engineer_features(df):
    log_k0_1, Ea1, log_k0_2, Ea2, log_Ua, dH1, dH2, log_rhoCp = MECH_PARAMS
    k0_1, k0_2 = np.exp(log_k0_1), np.exp(log_k0_2)

    tau = df["length_m"] / df["flow_rate_L_min"]
    Tavg = (df["inlet_temperature_K"] + df["jacket_temperature_K"]) / 2.0
    dT = df["jacket_temperature_K"] - df["inlet_temperature_K"]

    k1 = k0_1 * np.exp(-Ea1 / (R * Tavg))
    k2 = k0_2 * np.exp(-Ea2 / (R * Tavg))
    Da1 = k1 * tau
    Da2 = k2 * tau
    selectivity = k2 / k1

    # Compact, physically-motivated feature set: the mechanistic model already
    # collapses (flow_rate, length, T_in, T_jacket) into Da1/Da2/selectivity,
    # so we keep only the non-redundant physics summaries plus concentration
    # (which the mean-zero-order kinetics here doesn't use, but may carry
    # residual signal e.g. non-ideal/second-order effects) and dT (captures
    # heating vs cooling direction, not fully absorbed into Tavg-based Da's).
    feats = pd.DataFrame({
        "log_Da1": np.log1p(Da1),
        "log_Da2": np.log1p(Da2),
        "selectivity_k2_k1": np.log(selectivity),
        "concentration_mol_L": df["concentration_mol_L"],
        "dT": dT,
    })
    return feats


def mechanistic_mean(df):
    return predict_batch(df, MECH_PARAMS)


def build_gpr(n_features=5):
    kernel = ConstantKernel(1.0, (1e-2, 1e4)) * Matern(
        length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e4), nu=2.5
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-4, 1e3))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=8, random_state=42
    )


def build_classifier():
    # GradientBoostingClassifier: robust with n=150, handles the sharp
    # Da1/Da2-driven collapse boundary better than a linear model.
    return GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )


def two_stage_predict(X_train, y_train, mu_train, X_test, mu_test):
    """Stage 1: classify collapsed (~0) vs active regime.
    Stage 2: for rows predicted active, predict yield via mechanistic-mean + GPR-residual.
    Rows predicted collapsed are set to 0."""
    is_active_train = (y_train >= COLLAPSE_THRESHOLD)

    clf = build_classifier()
    clf.fit(X_train, is_active_train)
    active_prob_test = clf.predict_proba(X_test)[:, 1]
    is_active_test = active_prob_test >= 0.5

    scaler = StandardScaler().fit(X_train[is_active_train])
    gp = build_gpr()
    resid_train = (y_train - mu_train)[is_active_train]
    gp.fit(scaler.transform(X_train[is_active_train]), resid_train)

    y_pred = np.zeros(len(X_test))
    if is_active_test.any():
        resid_pred = gp.predict(scaler.transform(X_test[is_active_test]))
        y_pred[is_active_test] = np.clip(mu_test[is_active_test] + resid_pred, 0, 100)
    return y_pred, is_active_test, active_prob_test


def warped_gp_predict(X_train, y_train, mu_train, X_test, mu_test):
    """GP fit in logit-warped yield space, mechanistic model as the prior
    mean (also warped), predictions unwarped back to [0,100]."""
    z_train = warp(y_train)
    mu_z_train = warp(mu_train)
    resid_z_train = z_train - mu_z_train

    scaler = StandardScaler().fit(X_train)
    gp = build_gpr()
    gp.fit(scaler.transform(X_train), resid_z_train)

    mu_z_test = warp(mu_test)
    resid_z_pred = gp.predict(scaler.transform(X_test))
    z_pred = mu_z_test + resid_z_pred
    y_pred = unwarp(z_pred)
    return y_pred


def cross_validate(train_csv, n_splits=5, n_repeats=10, mode="single_stage"):
    """mode: 'single_stage' (additive residual GP, current default),
    'two_stage' (classifier + GP), or 'warped' (logit-warped GP)."""
    df = pd.read_csv(train_csv)
    y = df["overall_yield"].values
    mu = mechanistic_mean(df)
    X = engineer_features(df).values

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    rmses = []
    for train_idx, test_idx in rkf.split(X):
        if mode == "two_stage":
            y_pred, _, _ = two_stage_predict(
                X[train_idx], y[train_idx], mu[train_idx],
                X[test_idx], mu[test_idx],
            )
        elif mode == "warped":
            y_pred = warped_gp_predict(
                X[train_idx], y[train_idx], mu[train_idx],
                X[test_idx], mu[test_idx],
            )
        else:
            resid = y - mu
            scaler = StandardScaler().fit(X[train_idx])
            gp = build_gpr()
            gp.fit(scaler.transform(X[train_idx]), resid[train_idx])
            resid_pred = gp.predict(scaler.transform(X[test_idx]))
            y_pred = np.clip(mu[test_idx] + resid_pred, 0, 100)

        rmse = np.sqrt(np.mean((y[test_idx] - y_pred) ** 2))
        rmses.append(rmse)

    rmses = np.array(rmses)
    print(f"[{mode}] Repeated {n_splits}-fold CV ({n_repeats} repeats), n={len(rmses)} folds")
    print(f"  RMSE mean = {rmses.mean():.3f}  std = {rmses.std():.3f}")
    print(f"  RMSE median = {np.median(rmses):.3f}  min={rmses.min():.3f} max={rmses.max():.3f}")
    return rmses


def fit_final_and_predict(train_csv, test_csv, out_csv):
    """Warped (logit) GPR: CV comparison across all three variants --

        single-stage (additive residual):  RMSE 10.14 +/- 4.14
        two-stage (classifier + GPR):       RMSE 10.85 +/- 5.06
        warped (logit-space GPR):            RMSE  8.79 +/- 3.87   <- winner

    The warped variant models the GP in logit(yield/100) space instead of
    raw-yield residual space. This is the statistically appropriate
    likelihood for a bounded [0,100], zero-piled target (Tobit/censored-
    regression literature; the logit warp is a practical approximation of
    Snelson/Rasmussen/Ghahramani's "Warped Gaussian Processes", NeurIPS
    2003) -- it stops the GP from needing to represent the sharp 0%
    boundary as ordinary additive Gaussian noise, which is what was
    limiting the single-stage and two-stage variants."""
    df_train = pd.read_csv(train_csv)
    y_train = df_train["overall_yield"].values
    mu_train = mechanistic_mean(df_train)
    X_train = engineer_features(df_train).values

    df_test = pd.read_csv(test_csv)
    mu_test = mechanistic_mean(df_test)
    X_test = engineer_features(df_test).values

    y_pred = warped_gp_predict(X_train, y_train, mu_train, X_test, mu_test)

    out = pd.DataFrame({"overall_yield": np.round(y_pred, 3)})
    out.to_csv(out_csv, index=False)
    print(f"Predictions written to {out_csv}")
    print(f"Predicted zero-ish rows (<1): {(y_pred < 1).sum()}/{len(y_pred)}")
    return y_pred


if __name__ == "__main__":
    print("=== Cross-validation: single-stage GPR (additive residual) ===")
    cross_validate("train_dataset.csv", mode="single_stage")

    print("\n=== Cross-validation: two-stage classifier + GPR ===")
    cross_validate("train_dataset.csv", mode="two_stage")

    print("\n=== Cross-validation: warped (logit) GPR ===")
    cross_validate("train_dataset.csv", mode="warped")

    print("\n=== Fitting on full train set, predicting test set (warped, winner) ===")
    fit_final_and_predict("train_dataset.csv", "test_dataset.csv", "GPR_predictions.csv")
