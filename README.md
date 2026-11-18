# ScentMatch

ScentMatch recommends perfumes.
- **New visitors** answer a 6-question quiz. Their answers become olfactory notes (vanilla, oud, bergamot…), and the app suggests perfumes that carry those notes.
- **Known users** get recommendations from **NLGCL**, a graph neural network (Xu et al., RecSys 2025) trained on Amazon reviews.

Every recommendation comes with a short reason, for example *"Because you chose oud, amber and leather"*.

## Team (Group 4)

| Member | Role |
|---|---|
| Lisa El Gabteni | Project Lead |
| Maxime Dutertre | Data Engineer |
| Mina Aired | ML Engineer (model) |
| Halima Ghazi | ML Engineer (training) |
| Ambrine Senoussi | Software Engineer (API and app) |

## Quick start

Requires Python 3.11.

```bash
git clone https://github.com/madmax0978/scentmatch.git && cd scentmatch
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

bash scripts/download_data.sh          # public data, about 4.2 GB
python -m scentmatch.data.pipeline     # builds the dataset, about 35 s
python -m scentmatch.train.train       # trains the model, about 1 min on a laptop

scripts/run_demo.sh                    # starts the API and the website
```

Then open **http://localhost:8501**.
- To skip training, run `SCENTMATCH_MODE=mock scripts/run_demo.sh`. This shows placeholder recommendations.
- To try the "Your collection" tab, click a demo profile or paste the user id `AEJI7MSPXQCTPRTWBSQYK4NU7AOQ`.

## How it works

1. **Data:** Amazon Reviews 2023, perfumes only. A rating of 4 or 5 stars counts as "liked". The result is 23,290 users, 6,098 perfumes and 97,713 likes.
2. **Notes:** a list of 699 notes from Parfumo (open data) is matched against each product description. 75% of the perfumes get at least one note.
3. **Split:** for each user, the last like goes to the test set, the one before goes to the validation set, and the rest is used for training.
4. **Model:** NLGCL learns from the user–perfume graph. On validation it reaches NDCG@10 = 0.062, against 0.036 for "most popular".
5. **Quiz:** the app ranks perfumes by how well their notes match the chosen notes (60%). It adds what the model says people with that taste also love (25%) and the Amazon rating (15%).
6. **API and app:** a FastAPI server serves the recommendations, and the Streamlit website shows them. Each "Love it" or "Not for me" click is saved in a SQLite database.

## Useful commands

```bash
pytest -q                              # run the tests
python -m scentmatch.eval.harness      # compare with random and most-popular baselines
```

API documentation: http://localhost:8000/docs (while the demo runs).

## Project layout

```
configs/default.yaml     settings (paths, model size, training)
src/scentmatch/data/     data pipeline and note vocabulary
src/scentmatch/model/    NLGCL model and graph
src/scentmatch/train/    training and recommendation logic
src/scentmatch/eval/     metrics and baselines
src/scentmatch/api/      FastAPI server, quiz, database
app/streamlit_app.py     website
scripts/                 data download, demo launcher
tests/                   automated tests
```

## Troubleshooting

`No module named 'scentmatch'`: this happens on macOS when the folder is synced by iCloud. Start the app with `scripts/run_demo.sh`, or put `PYTHONPATH=src` in front of your command.
