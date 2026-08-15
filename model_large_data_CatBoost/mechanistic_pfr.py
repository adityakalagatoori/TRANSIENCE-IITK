"""
Mechanistic PFR model for series reaction A -> B -> C.

Physics (mole balances):
    dCA/dtau = -k1(T) * CA
    dCB/dtau =  k1(T) * CA - k2(T) * CB
    k_i(T)   =  k0_i * exp(-Ea_i / (R*T))

Energy balance (Fogler, "Elements of Chemical Reaction Engineering",
Ch. 8, steady-state non-isothermal PFR with jacket cooling):

    dT/dtau = [ Ua*(Tj - T) + (-dH1)*r1 + (-dH2)*r2 ] / (rho*Cp)

    r1 = k1(T)*CA   (rate of A -> B, releases/absorbs heat dH1)
    r2 = k2(T)*CB   (rate of B -> C, releases/absorbs heat dH2)

The first version of this model only had the Ua*(Tj - T) heat-exchange
term (equivalent to assuming both reactions are athermal), which is a
simplification -- the standard non-isothermal PFR energy balance also
includes reaction-heat generation/consumption terms. Since the reactions
are exothermic or endothermic to an unknown degree, dH1, dH2 and the
lumped heat-capacity term (rho*Cp) are fit as effective parameters
alongside the kinetics, the same way Ua was already being fit.

Since reactor diameter/cross-section area is not given, tau (residence
time) is proportional to length_m / flow_rate_L_min up to a constant
absorbed into k0. Likewise Ua and (rho*Cp) are effective lumped
parameters (not independently known), consistent with treating this as a
calibrated grey-box / semi-empirical model rather than a first-principles
simulation with known physical constants.

Effective parameters (k0_1, Ea1, k0_2, Ea2, Ua, dH1, dH2, rhoCp) are fit
to the training data via nonlinear least squares (scipy least_squares,
multi-start to avoid local optima).
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R = 8.314  # J/mol/K

N_PARAMS = 8  # log_k0_1, Ea1, log_k0_2, Ea2, log_Ua, dH1, dH2, log_rhoCp

# Fixed-step RK4 with a fixed step count: during parameter-space
# exploration, least_squares' finite-difference Jacobian probes corners
# of the (k0, Ea, dH, rhoCp) bounds where the ODE becomes extremely
# stiff (near-instantaneous CA depletion + large thermal derivatives).
# scipy's adaptive solvers (LSODA/Radau/RK23) all stalled on specific
# corners even with per-derivative clipping -- each fixed a symptom but
# a new stiffness mode kept appearing (traced through several rounds of
# isolating exact hanging parameter/row combinations). A fixed-step
# integrator has no adaptive step-size feedback loop to degenerate, so
# runtime is provably bounded (N_STEPS iterations, O(1) work each) for
# ANY parameter values, which is what the optimizer needs to be able to
# safely explore the space. N_STEPS=50 with RK4 (4th-order local
# accuracy) resolves tau ranges up to ~4.5 (this dataset's max) more
# than finely enough for the lumped-parameter calibration this model is
# doing. Lowered from an initial 200 after profiling showed 200 steps
# made a full multi-restart fit take ~90 minutes in pure-Python RK4;
# since RK4 error scales as O(h^4), 50 steps is still far more accurate
# than needed for fitting effective/lumped parameters (not claiming
# ODE-solver-grade precision, just a stable physics-informed surrogate).
N_STEPS = 50


def simulate_yield(flow_rate, conc0, T_in, length, T_jacket, params):
    """Integrate the non-isothermal PFR mole + energy balance for one
    operating condition using fixed-step RK4 (see N_STEPS comment above
    for why fixed-step instead of scipy's adaptive solvers).

    tau_max is proportional to length / flow_rate (residence time proxy);
    the proportionality constant is folded into k0_1, k0_2 during fitting.
    """
    log_k0_1, Ea1, log_k0_2, Ea2, log_Ua, dH1, dH2, log_rhoCp = params
    k0_1 = np.exp(log_k0_1)
    k0_2 = np.exp(log_k0_2)
    Ua = np.exp(log_Ua)
    rhoCp = np.exp(log_rhoCp)

    tau_max = length / flow_rate  # residence-time proxy (constant absorbed into k0's)

    T_CAP = 1500.0     # generous upper bound on plausible reactor temperature (K)
    RATE_CAP = 1e3      # mol/(L*tau) -- caps reaction rate magnitude
    DT_CAP = 1e4         # K/tau -- caps temperature-derivative magnitude

    def deriv(CA, CB, T):
        T = min(max(T, 1.0), T_CAP)
        CA = max(CA, 0.0)
        CB = max(CB, 0.0)
        k1 = k0_1 * np.exp(-Ea1 / (R * T))
        k2 = k0_2 * np.exp(-Ea2 / (R * T))
        r1 = min(k1 * CA, RATE_CAP)
        r2 = min(k2 * CB, RATE_CAP)
        dCA = -r1
        dCB = r1 - r2
        dT = (Ua * (T_jacket - T) + (-dH1) * r1 + (-dH2) * r2) / rhoCp
        dT = float(np.clip(dT, -DT_CAP, DT_CAP))
        return dCA, dCB, dT

    CA, CB, T = conc0, 0.0, T_in
    if tau_max > 0:
        h = tau_max / N_STEPS
        for _ in range(N_STEPS):
            k1a, k1b, k1c = deriv(CA, CB, T)
            k2a, k2b, k2c = deriv(CA + h/2*k1a, CB + h/2*k1b, T + h/2*k1c)
            k3a, k3b, k3c = deriv(CA + h/2*k2a, CB + h/2*k2b, T + h/2*k2c)
            k4a, k4b, k4c = deriv(CA + h*k3a, CB + h*k3b, T + h*k3c)
            CA += (h/6) * (k1a + 2*k2a + 2*k3a + k4a)
            CB += (h/6) * (k1b + 2*k2b + 2*k3b + k4b)
            T += (h/6) * (k1c + 2*k2c + 2*k3c + k4c)
            CA = max(CA, 0.0)
            CB = max(CB, 0.0)

    yield_pct = 100.0 * CB / conc0
    return float(np.clip(yield_pct, 0.0, 100.0))


def predict_batch(df, params):
    preds = np.zeros(len(df))
    for i, row in enumerate(df.itertuples()):
        preds[i] = simulate_yield(
            row.flow_rate_L_min, row.concentration_mol_L,
            row.inlet_temperature_K, row.length_m, row.jacket_temperature_K,
            params,
        )
    return preds


def residuals(params, df, y_true):
    y_pred = predict_batch(df, params)
    return y_pred - y_true


def fit_mechanistic_model(train_csv, n_restarts=6, seed=0):
    """Multi-start nonlinear least squares: the ODE-fit landscape is
    non-convex, so a single start can land in a poor local optimum.
    We try several randomized initial guesses and keep the best."""
    df = pd.read_csv(train_csv)
    y_true = df["overall_yield"].values

    # bounds: log_k0_1, Ea1, log_k0_2, Ea2, log_Ua, dH1, dH2, log_rhoCp
    # k0 upper bound tightened from 1e15 to 1e12: typical Arrhenius
    # pre-exponential factors for elementary reactions are 1e6-1e13; 1e15
    # combined with a small Ea let k(T) explode, which together with a
    # small rhoCp drove the ODE into thermal runaway and stalled the
    # stiff solver during least_squares' Jacobian finite-differencing
    # (observed in practice -- two fits hung and had to be killed).
    # rhoCp lower bound raised from 1e-2 to 1.0 for the same reason.
    bounds_lo = [np.log(1e2), 10000, np.log(1e2), 10000, np.log(1e-3), -1e5, -1e5, np.log(1.0)]
    bounds_hi = [np.log(1e12), 150000, np.log(1e12), 150000, np.log(1e3), 1e5, 1e5, np.log(1e4)]

    rng = np.random.default_rng(seed)
    starts = [[np.log(1e6), 50000, np.log(1e8), 70000, np.log(0.5), 0.0, 0.0, np.log(10.0)]]
    for _ in range(n_restarts - 1):
        starts.append([rng.uniform(lo, hi) for lo, hi in zip(bounds_lo, bounds_hi)])

    best_result = None
    for i, x0 in enumerate(starts):
        result = least_squares(
            residuals, x0, args=(df, y_true),
            bounds=(bounds_lo, bounds_hi),
            method="trf", verbose=0, max_nfev=200,
        )
        rmse = np.sqrt(2 * result.cost / len(y_true))
        print(f"  restart {i}: RMSE={rmse:.3f}")
        if best_result is None or result.cost < best_result.cost:
            best_result = result
    return best_result


if __name__ == "__main__":
    result = fit_mechanistic_model("train_dataset.csv")
    log_k0_1, Ea1, log_k0_2, Ea2, log_Ua, dH1, dH2, log_rhoCp = result.x
    print("\nFitted mechanistic parameters:")
    print(f"  k0_1   = {np.exp(log_k0_1):.4e}  Ea1 = {Ea1:.1f} J/mol")
    print(f"  k0_2   = {np.exp(log_k0_2):.4e}  Ea2 = {Ea2:.1f} J/mol")
    print(f"  Ua     = {np.exp(log_Ua):.4e}")
    print(f"  dH1    = {dH1:.1f} J/mol  (rxn 1 heat, A->B)")
    print(f"  dH2    = {dH2:.1f} J/mol  (rxn 2 heat, B->C)")
    print(f"  rhoCp  = {np.exp(log_rhoCp):.4e}")

    df = pd.read_csv("train_dataset.csv")
    y_true = df["overall_yield"].values
    y_pred = predict_batch(df, result.x)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"\nMechanistic model train RMSE: {rmse:.3f}")

    np.save("mechanistic_params.npy", result.x)
