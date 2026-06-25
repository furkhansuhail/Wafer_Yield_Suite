# Semiconductor Yield Modeling on WM-811K

Fit and compare three classical **analytical yield models** — Poisson, Murphy,
and Negative Binomial — on the WM-811K wafer-map dataset, then use the best fit
to **predict die yield as a function of area**:

> give an area **A** → get back the yield **Y** (the fraction of dies of that
> size expected to come out defect-free).

Fitting also produces interpretable process analytics: the defect density `D0`
and the Negative-Binomial clustering parameter `alpha`.

---

## Install

```bash
pip install -r requirements.txt
```

Then put your WM-811K pickle (`LSWMD.pkl`) in the project root (it is not
shipped with this code).

## Quickstart — train once, predict many times

```bash
# 1. TRAIN all three models, rank them, and save to models.json
python -m wm811k_yield.main train --data LSWMD.pkl --save models.json

# 2. PREDICT: area A -> yield Y
python -m wm811k_yield.main predict --load models.json --area 4
python -m wm811k_yield.main predict --load models.json --area 1 4 9 16

# inverse: largest die that still meets a target yield
python -m wm811k_yield.main predict --load models.json --target-yield 0.90
```

Example forward-prediction output:

```
    area       poisson        murphy   negative_binomial
    1.00        95.91%        95.33%        94.27% *
    4.00        84.61%        82.77%        79.83% *
    9.00        68.66%        65.92%        62.36% *
   16.00        51.25%        48.70%        46.33% *
( * = best-fitting model )
```

## Verify without the dataset

```bash
python tests/test_synthetic.py      # parameter recovery + correct model selection
python tests/test_persistence.py    # train -> save -> load -> predict round-trip
```

---

## The three models

All express yield as a function of `lambda = D0 * A`, differing only in how much
defect **clustering** they allow. They are nested: as `alpha -> inf`, Negative
Binomial collapses to Poisson.

| Model             | Formula                            | Clustering        |
|-------------------|------------------------------------|-------------------|
| Poisson           | `Y = exp(-D0*A)`                  | none (uniform)    |
| Murphy            | `Y = [(1-exp(-D0*A))/(D0*A)]^2`  | mild (triangular) |
| Negative Binomial | `Y = (1 + D0*A/alpha)^(-alpha)`  | free param alpha  |

Real wafers cluster (Center / Edge-Ring / Scratch patterns), so NB with small
`alpha` usually wins. The synthetic test confirms this: random defects -> Poisson
wins and `alpha` blows up; clustered defects -> NB wins with small `alpha`.

## How yield-vs-area is measured (Stapper window method)

Each wafer map encodes `0 = no die, 1 = good, 2 = defective`. For a window of
`w x w` dies (area `A = w^2`), a window *yields* only if every die inside is
good. Pooling windows over many wafers gives the empirical `Y(A)` curve the
models are fitted to (weighted least squares, weights from binomial std error).

## Project structure

```
wm811k-yield-modeling/
├── README.md                 this file
├── requirements.txt
├── wm811k_yield/             the package (one module = one job)
│   ├── config.py             Config dataclass — all tunables
│   ├── data_loader.py        WM811KLoader   — read pickle, clean, filter
│   ├── features.py           WindowExtractor — wafer maps -> YieldCurve
│   ├── eda.py                YieldEDA       — distributions, overdispersion, plots
│   ├── models/               PoissonYield / MurphyYield / NegativeBinomialYield
│   ├── trainer.py            YieldModelTrainer — fit model(s) to a curve
│   ├── evaluator.py          ModelEvaluator    — RMSE / R^2 / AIC / BIC leaderboard
│   ├── predictor.py          YieldPredictor    — yield(A), max area for target
│   ├── persistence.py        ModelStore        — save/load fitted models (JSON)
│   ├── main.py               train / predict commands
│   └── run.py                full pipeline with EDA plots
└── tests/
    ├── test_synthetic.py
    └── test_persistence.py
```

## A note on interpretation

A wafer map records **die-level** pass/fail on a fixed grid, not individual
physical point-defect coordinates. So `D0` here is a *failing-die* density per
unit-die area — a standard, reasonable proxy. State this explicitly when
reporting results.

## What these models do (and don't)

They predict the **yield rate** for a given die size — a population statistic.
They do **not** tell you which specific die fails or where (that is a
classification task, e.g. a CNN on wafer maps or sensor models on SECOM). Use
these when the goal is to characterize the defect process and forecast yield
for a proposed die area.
```
