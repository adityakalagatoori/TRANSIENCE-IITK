# Kshanik (क्षणिक) — Team Transience

**Physics-informed yield surrogate for a non-isothermal plug-flow reactor**
Fugacity 2026 ML Hackathon · IIT Kharagpur

Live brief: **[GitHub Pages site](https://adityakalagatoori.github.io/TRANSIENCE-IITK/)**

Team: Preenithi Varshana · Aditya Kalagatoori · Kritika Pandey

---

## 1. Problem

The competition ([`ML_Hackathon_Problem_Statement_Final.pdf`](ML_Hackathon_Problem_Statement_Final.pdf)) asks for a
data-driven surrogate that predicts `overall_yield` of a non-isothermal, continuous-flow reactor running the
series-parallel reaction network:

```
A --k1--> B --k2--> C      (B is the desired product)
```

given five operating conditions: `flow_rate_L_min`, `concentration_mol_L`, `inlet_temperature_K`, `length_m`,
`jacket_temperature_K`. Evaluation is RMSE on 50 held-out rows.

**Why this is hard with only 150 training rows:** yield is not a smooth function of the raw inputs. 25% of
training rows have `overall_yield == 0` (the side reaction B→C has fully consumed the product), and the
strongest single raw-feature correlation with yield is only `|r| = 0.64`. A black-box regressor has to infer a
sharp, non-linear collapse boundary from 150 points with no prior structure to lean on.

## 2. Approach

Instead of fitting ML directly to the raw inputs, we first fit the **actual reactor physics** — a non-isothermal
PFR ODE system (Arrhenius kinetics for both reactions, coupled to a jacket energy balance with explicit
reaction-heat terms, per Fogler's *Elements of Chemical Reaction Engineering*, Ch. 8) — via nonlinear least
squares on the 150 training rows. This mechanistic model's prediction then becomes a **prior mean** that the ML
models only need to correct, not replace.

Two models are built for two different data regimes:

| | Small-data model | Large-data model |
|---|---|---|
| **Folder** | [`model_small_data_GPR/`](model_small_data_GPR/) | [`model_large_data_CatBoost/`](model_large_data_CatBoost/) |
| **Model** | Warped (logit-space) Gaussian Process Regression | CatBoost, two-stage classifier + regressor |
| **Data used** | 150 real rows only | 150 real rows + 5,000 ODE-simulated synthetic rows |
| **CV RMSE** | **8.79** | **11.35** |
| **When to use** | This competition (n=150) | If the dataset scales up in production |

The warped-GP is the submission candidate — the competition only allows one final submission, and it has the
lower cross-validated RMSE.

## 3. Repository layout

```
.
├── ML_Hackathon_Problem_Statement_Final.pdf   Original problem statement (reference only)
├── README.md                                  This file
├── docs/
│   └── index.html                             Project brief — served via GitHub Pages (see §6)
├── audio/                                      Audio walkthroughs / pitch recordings (added separately)
├── model_small_data_GPR/                       SMALL-DATA MODEL — physics-informed GPR
│   ├── train_dataset.csv                       150 labeled rows (competition-provided)
│   ├── test_dataset.csv                        50 unlabeled rows to predict (competition-provided)
│   ├── mechanistic_pfr.py                      Non-isothermal PFR ODE model + least-squares fitter
│   ├── mechanistic_params.npy                  Saved fitted kinetic/thermal parameters (output of mechanistic_pfr.py)
│   ├── gpr_model.py                            Feature engineering, warped-GP training, CV, final predictions
│   └── GPR_predictions.csv                     FINAL submission-candidate predictions (50 rows, overall_yield)
└── model_large_data_CatBoost/                  LARGE-DATA MODEL — CatBoost + synthetic augmentation
    ├── train_dataset.csv                       Same 150 rows (copied locally so this folder is self-contained)
    ├── test_dataset.csv                        Same 50 rows
    ├── mechanistic_pfr.py                      Same ODE model, copied so this folder runs independently
    ├── mechanistic_params.npy                  Same fitted parameters, copied from the GPR folder
    ├── generate_synthetic.py                   Latin Hypercube sampling + ODE simulation → synthetic_dataset.csv
    ├── synthetic_dataset.csv                   5,000 physically-consistent synthetic rows (output of generate_synthetic.py)
    ├── catboost_model.py                       Feature engineering, two-stage classifier+regressor, CV, final predictions
    └── CatBoost_predictions.csv                Alternate predictions (50 rows, overall_yield)
```

**Why two copies of `mechanistic_pfr.py` / `mechanistic_params.npy`:** each model folder is deliberately
self-contained — either one can be zipped up and run on its own without needing the sibling folder.

## 4. File-by-file: what each script does

### `model_small_data_GPR/mechanistic_pfr.py` (shared foundation)
Defines the ODE system (`dCA/dτ`, `dCB/dτ`, `dT/dτ`) and integrates it with a fixed-step RK4 scheme
(`simulate_yield`). `fit_mechanistic_model()` runs multi-start `scipy.optimize.least_squares` to fit 8 effective
parameters — `k0_1, Ea1, k0_2, Ea2` (Arrhenius kinetics for both reactions), `Ua` (jacket heat-transfer
coefficient), `dH1, dH2` (reaction enthalpies), `rhoCp` (lumped heat capacity) — against `train_dataset.csv`.
Running it as a script re-fits from scratch and overwrites `mechanistic_params.npy`.

### `model_small_data_GPR/gpr_model.py`
1. `engineer_features()` — derives physics features (Damköhler numbers `Da1`/`Da2`, selectivity ratio,
   residence time, ΔT) from the fitted mechanistic parameters.
2. `warped_gp_predict()` — the core method: trains a Matérn-kernel GP in **logit(yield/100) space** (not raw
   yield), with the mechanistic model's own prediction as the GP's prior mean, and un-warps predictions back
   to `[0, 100]`.
3. `cross_validate()` — repeated 5-fold CV (10 repeats) comparing three variants: single-stage additive
   residual, two-stage classifier+GP, and the warped-GP (winner).
4. `fit_final_and_predict()` — fits on the full 150 rows and writes `GPR_predictions.csv`.

### `model_large_data_CatBoost/generate_synthetic.py`
Samples the 5 input ranges (with a 5% margin) via Latin Hypercube Sampling, runs each sample through the fitted
ODE (`mechanistic_pfr.simulate_yield`) to generate 5,000 physically-consistent `(features, yield)` synthetic
rows, saved to `synthetic_dataset.csv`.

### `model_large_data_CatBoost/catboost_model.py`
Same physics-feature engineering as the GPR model. `two_stage_predict()` trains a CatBoost classifier to detect
the collapsed-yield regime, then a CatBoost regressor (real rows weight 1.0, synthetic rows weight 0.15) for the
active regime. `cross_validate_real_only()`, `cross_validate_with_synthetic()`, and
`cross_validate_two_stage()` compare all three configurations on real data only (synthetic rows never enter a
held-out fold, to avoid circular validation). `fit_final_and_predict()` writes `CatBoost_predictions.csv`.

## 5. How to run

### Setup
```bash
python -m pip install numpy pandas scipy scikit-learn catboost
```

### Small-data model (GPR) — the submission candidate
```bash
cd model_small_data_GPR

# 1. (Optional) Re-fit the mechanistic ODE from scratch — only needed if train_dataset.csv changes.
#    Takes ~15 minutes (multi-start least squares over a stiff ODE system).
python mechanistic_pfr.py

# 2. Run cross-validation across all GPR variants and generate final test-set predictions.
#    Takes ~1-2 minutes. Overwrites GPR_predictions.csv.
python gpr_model.py
```

### Large-data model (CatBoost)
```bash
cd model_large_data_CatBoost

# 1. (Optional) Re-fit the mechanistic ODE — skip if mechanistic_params.npy is already current
#    (copy it from model_small_data_GPR/ after running step 1 above, or re-run mechanistic_pfr.py here).
python mechanistic_pfr.py

# 2. Regenerate the synthetic dataset from the current mechanistic fit.
python generate_synthetic.py

# 3. Run cross-validation (real-only, real+synthetic, two-stage) and generate final predictions.
#    Takes several minutes — trains multiple CatBoost models per CV fold.
python catboost_model.py
```

Both `*_model.py` scripts print cross-validation RMSE to the console before writing their prediction CSV, so you
can confirm the numbers in §7 reproduce before trusting the output file.

## 6. GitHub Pages (project brief)

[`docs/index.html`](docs/index.html) is a self-contained, single-file HTML brief (no build step, no external
dependencies) summarizing the problem, approach, and results in the same visual language across both models.
GitHub Pages serves it directly from the `docs/` folder on the default branch — see the exact repository
settings step in the commands below.

## 7. Results summary

| Model variant | CV RMSE | Notes |
|---|---|---|
| Mechanistic ODE alone (no ML) | 12.17 | Physics-only baseline |
| GPR — additive residual | 10.14 | Plain GP on raw residual |
| **GPR — warped (logit) residual** | **8.79** | **Selected small-data model** |
| CatBoost — real data only | 15.24 | No synthetic augmentation |
| CatBoost — real + synthetic | 12.67 | Synthetic augmentation, single-stage |
| **CatBoost — two-stage, real + synthetic** | **11.35** | **Selected large-data model** |

All RMSE values are 5-fold cross-validation, repeated 10×, evaluated on real training rows only.

## 8. References

- H. S. Fogler, *Elements of Chemical Reaction Engineering*, Ch. 8 — non-isothermal PFR energy balance.
- R. Kiryo, G. Niu, M. C. du Plessis, M. Sugiyama, "Positive-Unlabeled Learning with Non-Negative Risk
  Estimator," NeurIPS 2017 — background reading on label-scarce learning (referenced during method selection).
- E. Snelson, C. E. Rasmussen, Z. Ghahramani, "Warped Gaussian Processes," NeurIPS 2003 — basis for the
  logit-warped GP used in the small-data model.
