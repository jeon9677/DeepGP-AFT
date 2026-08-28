# DeepGP-AFT

Simulation code for **A Deep Gaussian Process for Survival Prediction under the
Accelerated Failure Time Model**.

The repository provides a reproducible two-step workflow:

1. generate right-censored log-normal AFT simulation data;
2. train the DeepGP-AFT approximation and collect seed-level metrics.

## Repository layout

```text
DeepGP-AFT/
├── Simulation/
│   ├── generate_data.py   # manuscript simulation data generator
│   └── deepgp_aft.py      # model, MC-dropout inference, and evaluation
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/jeon9677/DeepGP-AFT.git
cd DeepGP-AFT
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Generate simulation data

The manuscript generates

$$
X_i\sim N_p(0,\Sigma), \qquad \Sigma_{jk}=0.15^{|j-k|},
$$

$$
\log T_i=g(X_i)+\epsilon_i,\qquad
\epsilon_i\sim N(0,\sigma_\epsilon^2),
$$

where

$$
g(x)=x_1x_2+0.5x_3^2+\sin(\pi x_4/2)-0.8x_5
     +\sum_{j=6}^{p}w_jx_j^3,
$$

and the weights decrease linearly from 0.2 to -0.1. Censoring times follow
$C_i \sim \mathrm{Uniform}(0, \tau)$, with
$Y_i = \min(T_i, C_i)$ and
$\delta_i = \mathbf{1}(T_i \le C_i)$.

Example: generate 100 replicates for one scenario (seeds 1000-1099).

```bash
python Simulation/generate_data.py \
  --p 30 \
  --error-variance 0.25 \
  --tau 6.0 \
  --replicates 100 \
  --first-seed 1000 \
  --outdir simul_AFT/SimulData
```

This creates:

```text
simul_AFT/SimulData/n1000_p30_sigma0.25_tau6.0/
├── seed_1000_train.csv
├── seed_1000_test.csv
├── ...
├── seed_1099_train.csv
├── seed_1099_test.csv
└── simulation_metadata.csv
```

Each CSV contains `x1,...,xp`, observed time `y`, event indicator `delta`, and
simulation-only oracle columns `logT`, `mu_true`, and `sigma_true`. Oracle
columns are never used as model inputs; `logT` is used only for simulation
RMSE and coverage evaluation.

## 2. Train and evaluate DeepGP-AFT

The implementation uses a ReLU feed-forward representation of the DeepGP mean
function, hidden-layer dropout during training and prediction, the right-censored
log-normal likelihood, and one global residual scale. The default architecture is:

```text
Input(p)
  -> Dense(128, ReLU) -> Dropout(0.2)
  -> Dense(128, ReLU) -> Dropout(0.2)
  -> Dense(64,  ReLU) -> Dropout(0.2)
  -> Dense(1, linear): mu(x)
  +  one trainable global sigma_epsilon
```

Weights and biases receive the dropout-variational L2 penalty described in the
manuscript. At prediction, 100 stochastic dropout passes estimate epistemic
variance. Aleatoric variance is replaced by the mean squared residual among
uncensored training subjects, as specified in the manuscript.

Run every `n*` scenario directory under `SimulData`:

```bash
python Simulation/deepgp_aft.py \
  --root simul_AFT/SimulData \
  --widths 128,128,64 \
  --dropout 0.2 \
  --batch-size 64 \
  --epochs 300 \
  --patience 20 \
  --mc-samples 100 \
  --require-all-seeds
```

Run only one scenario:

```bash
python Simulation/deepgp_aft.py \
  --dir simul_AFT/SimulData/n1000_p30_sigma0.25_tau6.0
```

`--root` and `--dir` are mutually exclusive: use exactly one.

## Output

Each scenario directory receives one checkpointed file named
`summary_updateAFT.csv`, with one row per completed seed:

| Column | Definition |
|---|---|
| `seed` | simulation seed |
| `rmse` | full-test RMSE of predicted versus true `logT` |
| `ipcw` | IPCW C-index with pair weight $\delta_i/\hat G(Y_i)^2$ |
| `coverage` | empirical coverage of the 95% predictive interval for `logT` |
| `interval` | mean 95% predictive-interval width on the log-time scale |
| `epochs_trained` | epochs actually trained before early stopping |
| `runtime_seconds` | fitting and prediction time for the seed |

The file is updated after every seed, so completed results remain available if a
long run is interrupted.

## Default training settings

| Option | Default |
|---|---:|
| hidden widths | `128,128,64` |
| dropout | `0.2` |
| batch size | `64` |
| maximum epochs | `300` |
| early-stopping patience | `20` |
| Adam learning rate | `0.001` |
| MC-dropout samples | `100` |
| validation fraction | `0.2` |
| seed range | `1000-1099` |

Use `python Simulation/generate_data.py --help` and
`python Simulation/deepgp_aft.py --help` for the complete CLI reference.

## Reproducibility notes

- `sigma` in directory names denotes the Gaussian **error variance**, not its
  standard deviation.
- TensorFlow operations can show small platform-dependent numerical differences.
- Architecture selection should be performed separately from final test-set
  evaluation. Do not select hyperparameters using all 100 test sets and then
  report those same test sets as an unbiased final comparison.

## Citation

If you use this repository, please cite the associated DeepGP-AFT manuscript.
