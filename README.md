# ScentMatch

ScentMatch (Group 4) recommends perfumes from the notes a user loves. It uses **NLGCL** graph contrastive learning (Xu et al., ACM RecSys 2025) over a user–perfume(–note) graph. A short taste quiz serves brand-new users from their first visit.

| Member | Role | Owns |
|---|---|---|
| Lisa El Gabteni | Project Lead (PL) | evaluation harness, split, graph construction |
| Maxime Dutertre | Data Engineer (DE) | data pipeline, note vocabulary, SQLite, `/log_interaction` |
| Mina Aired | Lead ML Engineer (MLE-Core) | NLGCL model, note-node extension |
| Halima Ghazi | ML Engineer (MLE-Ops) | training loop, checkpoints, `predict()` |
| Ambrine Senoussi | Systems Engineer (SE) | FastAPI backend, Streamlit app, deployment |

## Status (Week 2)
The data pipeline runs on the full dataset. The evaluation harness, the NLGCL model and training, and the API work end to end. The API serves mock recommendations by default; `SCENTMATCH_MODE=model` makes it serve the trained model. See `reports/` for the status report.

## Run locally (about 5 minutes after the data download)

```bash
# 1. clone and enter the repo
git clone https://github.com/madmax0978/scentmatch.git && cd scentmatch

# 2. create a virtual environment (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate

# 3. install
pip install -r requirements.txt && pip install -e .

# 4. download the raw data (about 4.2 GB, public)
bash scripts/download_data.sh

# 5. build the dataset: fragrance filter, notes, k-core, split, SQLite (about 35 s)
python -m scentmatch.data.pipeline

# 6. start the API (mock mode; use SCENTMATCH_MODE=model after step 8)
uvicorn scentmatch.api.main:app --port 8000
#    -> http://localhost:8000/docs

# 7. in a second terminal, start the app
streamlit run app/streamlit_app.py
#    -> http://localhost:8501
```

### Train the model (optional, about 40 s on a laptop CPU)
```bash
python -m scentmatch.train.train                 # user–perfume graph -> checkpoints/nlgcl.pt
python -m scentmatch.train.train --use-notes     # user–perfume–note graph
python -m scentmatch.train.train --forward-only  # smoke test: one forward pass
SCENTMATCH_MODE=model uvicorn scentmatch.api.main:app --port 8000
```

### Evaluate
```bash
python -m scentmatch.eval.harness   # random + most-popular sanity floor -> reports/baselines.json
pytest -q                           # metrics on mock predictions, model forward pass, API contract
```

## Data
- **Interactions:** Amazon Reviews 2023, `Beauty_and_Personal_Care`, rating-only file. We keep items whose categories contain "Fragrance". Ratings ≥ 4 count as positives, with an iterative 3-core and a leave-last-out temporal split.
- **Note vocabulary:** Parfumo open data (TidyTuesday 2024-12-10). It provides 699 notes and 16 accords, matched in each perfume's title, description, features and `Scent` field.
- **New interactions:** `POST /log_interaction` writes to `data/scentmatch.db`, table `interactions`.

## Layout
```
configs/default.yaml        hyper-parameters and paths
src/scentmatch/data/        vocab.py, pipeline.py
src/scentmatch/eval/        metrics.py, harness.py, baselines.py
src/scentmatch/model/       graph.py, nlgcl.py
src/scentmatch/train/       train.py, predict.py (mock + model recommenders)
src/scentmatch/api/         main.py (FastAPI), db.py (SQLite), quiz.py (quiz -> notes)
app/streamlit_app.py        front end
tests/                      pytest suite
```
