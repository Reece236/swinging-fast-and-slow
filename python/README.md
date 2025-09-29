Python replication of "Swinging, Fast and Slow"

Prereqs
- Data: copy `data/` from the DOI into `/workspace/data` (see repository `README.md` for the DOI: `https://doi.org/10.25611/7QXV-8612`).
- Models: copy `models/` from the DOI into `/workspace/models`.
- Python 3.10+ with a working compiler toolchain (PyMC uses JAX/NumPy backend).

Install
```bash
python -m venv /workspace/python/.venv
/workspace/python/.venv/bin/pip install -U pip
/workspace/python/.venv/bin/pip install -r /workspace/python/requirements.txt
```

Run
```bash
/workspace/python/.venv/bin/python /workspace/python/run_replication.py \
  --data_dir /workspace/data \
  --models_dir /workspace/models \
  --out_dir /workspace/python/out
```

Notes
- The R pipeline generates pitch and hit outcome predictions using `predpitchscore` (R) and an XGBoost model saved as `models/hit_outcome_model.rds`. This Python replication fits the Bayesian intention model and the causal GLMs; for the causal step, it expects a file with per-pitch predictions (e.g., `pred_outcome_pitch.csv`) or you can generate those with the original R code and export them for Python to consume.
- The intention model mirrors the BRMS skew-normal hierarchical model using PyMC. Sampling can be time-consuming; reduce `draws`/`tune` in `run_replication.py` to prototype.
