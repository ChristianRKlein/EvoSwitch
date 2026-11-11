# EvoSwitch

**Timing and patient selection for strategic interruptions of targeted cancer therapy**

This repository contains the code and numerical results accompanying

> Klein, C. R. & Brossart, P. *Optimal Initiation and Clonal Competition in Targeted Cancer Therapy Interruptions.* Manuscript submitted to *npj Systems Biology and Applications* (2026).

The study uses a minimal model of competing drug-sensitive and drug-resistant cancer cells to ask three clinical questions: when a strategic treatment interruption should start, whether a good early response is a sufficient reason to interrupt, and how much pre-existing resistance an interruption can tolerate. All results are deterministic simulations; **no patient data were used**.

## Model

Sensitive cells `s` and resistant cells `r` (scaled to a common carrying capacity) follow

```
ds/dt = g_S s (1 - s - r) - kappa u s - nu s
dr/dt = g_R r (1 - r - alpha s) + nu s
```

with treatment status `u = 1` (drug given) or `u = 0` (interruption). The drug kills only sensitive cells; resistant cells carry a fitness cost (`g_R < g_S`), are restrained by sensitive cells through competition (`alpha`) and benefit from therapy only indirectly, through competitive release. Sensitive cells convert to resistance at the small rate `nu`. Time is measured in model days.

The main endpoint is the time to progression gained or lost through one interruption,
`Delta T = T_interruption - T_continuous`, where progression is the first time the total burden `b = s + r` rises to a fixed threshold. Nadir-relative progression criteria and restart rules are analyzed in Supplementary Note 5.

## Repository structure

```
EvoSwitch/
├── evoswitch/                 Python package
│   ├── model.py               model equations, solver settings, simulation
│   ├── endpoints.py           response, progression and Delta T
│   ├── matching.py            response matching; resistant fraction compatible with a response
│   ├── sensitivity.py         start-time sensitivity (Eq. S4)
│   ├── nadir.py               nadir-relative criteria and restart rules
│   └── parameters.py          parameter settings of the manuscript (Table 3)
├── analyses/                  one script per analysis; each writes CSV files to results/
│   ├── 01_timing_example.py ... 10_convergence.py
│   └── make_figures.py        all figures from the CSV files
├── results/                   numerical results underlying every figure and table (CSV)
├── figures/                   figures as in the manuscript (PNG, 300 dpi)
├── tests/                     fast reproduction checks (pytest)
├── run_all.sh                 runs all analyses and creates all figures
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/ChristianRKlein/EvoSwitch.git
cd EvoSwitch
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The results were produced with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

## Quick start

```python
from evoswitch import TIMING_EXAMPLE, response_day, interruption_gain

p = TIMING_EXAMPLE                       # 1% pre-existing resistant cells, strong competition
day = response_day(p, remaining=0.7**3)  # RECIST-equivalent response (66% volume reduction)
gain, crossed = interruption_gain(p, start=126, length=28, b_crit=0.27)
print(f"response on day {day:.1f}; a 4-week interruption gains {gain:.1f} days")
# response on day 125.4; a 4-week interruption gains 46.1 days
```

## Reproducing the results

```bash
pytest -q                 # fast checks of key numbers (a few seconds)
bash run_all.sh           # all analyses and figures
```

The analyses can be run individually from the `analyses/` folder (for example `python 04_tumors_A_B.py`). Scripts with parameter scans run in parallel on all available cores; set `EVOSWITCH_WORKERS=1` to run them serially. On a laptop with two cores, the complete set of analyses takes about 10 minutes; the timing scan (`02`) is the longest step. The precomputed results are included in `results/`, so `python analyses/make_figures.py` recreates all figures within seconds.

| Manuscript item | Script | Output in `results/` |
|---|---|---|
| Fig. 1 (concept) | `make_figures.py` | – |
| Fig. 2; Suppl. Tables 6, 7; Suppl. Fig. 1 | `01_timing_example.py` | `timing_*.csv`, `start_time_sensitivity.csv` |
| Optimal start: systematic scan; Suppl. Table 9 | `02_timing_scan.py` | `timing_scan_1day.csv`, `timing_grid16.csv` |
| Optimal start and drug effect; Suppl. Table 8 | `03_drug_effect_scan.py` | `drug_effect_scan.csv`, `drug_effect_example.csv` |
| Table 1; Fig. 3a,b; Suppl. Tables 3–5 | `04_tumors_A_B.py` | `tumors_AB*.csv` |
| Fig. 3c (1,368 response-matched tumors) | `05_response_matched_map.py` | `response_matched_map.csv` |
| Fig. 4; Table 2; Suppl. Tables 10–12; Suppl. Fig. 2 | `06_resistance_limits.py` | `fixed_drug_effect_*.csv`, `limit_curves.csv`, `upper_limits.csv`, `lower_limit*.csv` |
| Resistant fraction compatible with a response; Suppl. Table 13 | `07_max_compatible_fraction.py` | `max_compatible_fraction.csv`, `no_upper_limit_check.csv` |
| Nadir-relative criteria and restart rules; Suppl. Table 14; Suppl. Fig. 3 | `08_nadir_criteria.py` | `nadir_examples.csv`, `restart_rules_map.csv` |
| Extreme scenarios; Suppl. Table 15 | `09_extreme_scenarios.py` | `extreme_scenarios.csv` |
| Numerical convergence; Suppl. Table 17 | `10_convergence.py` | `convergence.csv` |

Supplementary Tables 1, 2 and 16 contain no simulation results (model correspondence, analytical regimes and published evidence).

## Numerical methods

Equations are integrated with SciPy `solve_ivp` (LSODA; relative tolerance 1e-10, absolute tolerance 1e-13, maximum step 2 days). Treatment switches are integration boundaries; progression times, nadirs and restart triggers are located by event detection. Drug effects for response matching are obtained with Brent's method (tolerance 1e-14). All results are deterministic and unchanged to five decimal places under tightened solver settings (`10_convergence.py`).

## Scope and limitations

The model is a deliberately minimal, theoretical description. Its parameters are orders of magnitude taken from a published model of abiraterone-treated prostate cancer (Zhang et al., *eLife* 2022) and are not calibrated to a specific cancer or drug. Durations are model days, and numerical thresholds are illustrations, not clinical cut-offs. The code is intended for research and must not be used to guide the treatment of individual patients.

## Citation

If you use this code, please cite the accompanying article (see above) and this repository (`CITATION.cff`).

## License

MIT License (see `LICENSE`).

## Contact

Christian R. Klein, Clinic of Internal Medicine III, University Hospital Bonn, Germany — christian.klein@ukbonn.de
