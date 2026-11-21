# ScentMatch

ScentMatch recommends perfumes, and says why.
- **New visitors** answer a 6-question quiz. Their answers become olfactory notes (vanilla, oud, bergamot…), and the app suggests perfumes.
- **Every click counts.** "Love it" and "Not for me" are saved, and the selection is recomputed at once: more like what you loved, less like what you turned down. Changed your mind? Every choice can be undone or switched, from the selection or from the "Your collection" tab.
- **Known users** get recommendations from **NLGCL**, a graph neural network (Xu et al., RecSys 2025) trained on Amazon reviews.
- **Every recommendation is explained.** The score is a sum of visible parts: your notes, the perfumes you loved, what you rejected. Open "Why this perfume?" on any card.

No Amazon star rating is used to rank perfumes.

## Team (Group 4)

| Member | Role |
|---|---|
| Lisa El Gabteni | Project Lead |
| Maxime Dutertre | Data Engineer |
| Mina Aired | ML Engineer (model) |
| Halima Ghazi | ML Engineer (training) |
| Ambrine Senoussi | Systems Engineer (API and app) |

## Quick start (about 5 minutes)

Requires Python 3.11 (`brew install python@3.11` on macOS) and `git`.

```bash
git clone https://github.com/madmax0978/scentmatch.git && cd scentmatch
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e .
scripts/fetch_release.sh && scripts/run_demo.sh
```

Then open **http://localhost:8501**. `fetch_release.sh` downloads the prepared data and the trained model (about 30 MB) from the GitHub Release and checks their SHA-256.

- No release yet, or offline? `SCENTMATCH_MODE=mock scripts/run_demo.sh` shows placeholder recommendations.
- "Your collection" tab: your quiz, the perfumes you loved and turned down (undo or switch any of them), and below it the demo profiles of real Amazon clients (or paste the user id `AEJI7MSPXQCTPRTWBSQYK4NU7AOQ`).
- Your visit is kept in the link (`?visitor=web-…`): reload the page or come back with the same link and your choices are still there.
- API documentation: http://localhost:8000/docs while the demo runs.

## Rebuild everything from the raw data (about 1 hour)

```bash
bash scripts/download_data.sh                  # Amazon metadata + ratings (4.2 GB) and Parfumo notes
python -m scentmatch.data.pipeline             # fragrance filter, notes, 3-core, split, SQLite (about 1 min)
python -m scentmatch.data.reviews              # streams the 11 GB review file, keeps our 106k reviews (about 7 min)
python -m scentmatch.data.feedback             # reads the 1-3 star reviews: dislike / like / not about the scent
python -m scentmatch.data.coldstart            # user segments, cold perfumes, 2,000 held-out users
python -m scentmatch.train.train --run-name final_nlgcl_notes_rescued_s3   # the served model (tuned settings, about 3 min)
python -m scentmatch.train.train --train-file cold/train.parquet --run-name cold_nlgcl   # model for the cold-start test
pytest -q                                      # 30+ tests, including the leakage audit
```

Week 3 experiments (validation only, then the test split once):

```bash
python -m scentmatch.train.sweep core                       # contrastive weight x regularisation (reports/sweeps/)
python -m scentmatch.train.sweep final --jobs reports/sweeps/jobs/final.json   # 3 seeds per model
python -m scentmatch.eval.final_report                      # every model x 3 seeds, validation + test (once)
python -m scentmatch.eval.coldstart_eval --ckpt cold_nlgcl.pt --ckpt-notes cold_nlgcl_notes.pt   # quiz + like loop
python -m scentmatch.eval.feedback_eval --ckpt final_nlgcl_notes_rescued_s3.pt --split val       # "Not for me" penalty
python -m scentmatch.eval.analysis --ckpt final_nlgcl_notes_rescued_s3.pt --runs final_lightgcn_s3 final_nlgcl_s3 final_nlgcl_notes_rescued_s3   # figures
python scripts/benchmark_api.py                             # latency + SQLite proof (API must run)
```

## How it works

1. **Data.** Amazon Reviews 2023, perfumes only. A rating of 4 or 5 stars counts as "liked". After a 3-core: 23,290 users, 6,098 perfumes, 97,713 likes. For each user the last like is the test item, the one before the validation item, the rest is training.
2. **Low ratings are read, not thrown away.** A 1-star rating can mean "I hate this smell" or "the bottle arrived broken". We read the text of the 8,622 reviews with 1-3 stars and sort them with explainable keyword rules: dislikes the smell, likes the smell (but it does not last…), or not about the smell. Checked by hand on 80 held-out reviews: 85% of the "dislikes" and 90% of the "likes" are right.
3. **Notes.** 697 notes from Parfumo (open data) are matched in each product description (99% precision on a hand-checked sample of 288 matches). 75% of the perfumes get at least one note.
4. **Model.** NLGCL on the user–perfume(–note) graph, tuned on validation. Explicit dislikes from reviews can be used as hard negatives.
5. **Serving.** `score = w_notes × notes match + w_collab × taste similarity (NLGCL) − w_avoid × closeness to what you rejected`. The weights and the popularity knob β were tuned on 1,000 held-out users and checked on 1,000 others.
6. **App.** FastAPI serves the recommendations, Streamlit shows them. Each click is stored in SQLite (`interactions`), and `POST /recommendations/session` rebuilds the visitor's profile from it. The log is append-only: taking a choice back is an `undo` event, the latest event on a perfume wins, and `GET /session/{user_id}` returns the current quiz and choices.

Results, figures and the leakage audit: [`reports/WEEK3_RESULTS.md`](reports/WEEK3_RESULTS.md).

## Project layout

```
configs/default.yaml       settings (tuned values, served checkpoint)
src/scentmatch/data/       pipeline, note vocabulary, review texts, feedback labels, cold-start sets
src/scentmatch/model/      NLGCL model and graph
src/scentmatch/train/      training, sweeps, recommendation + explanation logic (predict.py)
src/scentmatch/eval/       metrics, harness, full evaluation, cold-start evaluation, analysis
src/scentmatch/api/        FastAPI server, quiz, SQLite
app/streamlit_app.py       website
scripts/                   data download, release, demo launcher, API benchmark
reports/                   results, sweeps, audits, figures
tests/                     automated tests (pytest)
```

## Troubleshooting

`No module named 'scentmatch'`: this happens on macOS when the folder is synced by iCloud. Start the app with `scripts/run_demo.sh`, or put `PYTHONPATH=src` in front of your command. Better: keep the repository outside iCloud (e.g. `~/code/scentmatch`).
