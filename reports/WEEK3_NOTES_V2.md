# ScentMatch — Week 3, notes v2: is the note graph worth it? (Mina)

Status on 22 Nov 2026. Everything below comes from `scripts/notes_ablation.sh`; the full tables are
`reports/week3_notes_ablation_val.md` (used for every decision) and `reports/week3_notes_ablation_test.md`
(read once, after the freeze). Known users: mean ± std over 3 seeds. Brand-new visitors: the 1,000-user
"test" half of the cold-start split, models trained without them, all three with the same training
config (seed 3, liked low-star reviews) and the same served Scorer weights.

## 1. Improving the notes version

| Lever | Tried (validation) | Result |
|---|---|---|
| Lighter perfume–note edges | w = 1, 0.3, 0.1 (`reports/sweeps/notes.csv`) | 0.3 kept (Week 3) |
| Note contrastive weight λ_note | 0, 1e-5, 5e-5 at every w; again at k = 3 (`mainnotes2.csv`) | 1e-5 with all notes; 0 with main notes |
| **Only the main notes of each perfume** (new) | k = 3, 5, 8 most distinctive notes (IDF) vs all (`mainnotes.csv`) | k = 3 best of the three |

Notes are extracted from Amazon text, so some listings carry 20–45 notes, mostly generic ones.
`--max-notes k` keeps the k rarest notes of each perfume as graph edges (31,076 → 11,890 edges for k = 3).

## 2. The evidence table — with notes vs without notes (TEST)

| Metric | No notes | All notes (frozen) | Main notes k = 3 |
|---|---|---|---|
| **Accuracy, known users** NDCG@10 | 0.0653 ± 0.0003 | **0.0662 ± 0.0004** | 0.0659 ± 0.0006 |
| Recall@20 | 0.1631 ± 0.0011 | 0.1649 ± 0.0007 | **0.1656 ± 0.0011** |
| **Cold-start, quiz only** NDCG@10 | **0.0493** | 0.0487 | 0.0369 |
| Cold-start, quiz only Recall@20 | 0.0854 | **0.0904** | 0.0600 |
| Cold-start, quiz + 1 like NDCG@10 | 0.0848 | **0.0862** | 0.0856 |
| **Coverage@10** (known users) | **45.1%** | 38.7% | 43.7% |
| Coverage@10, served β = 0.5 | **72.5%** | 65.3% | 71.7% |
| Coverage@10, quiz only | 24.3% | 24.0% | **29.2%** |
| **Diversity** note ILD@10 (known users) | **0.903** | 0.899 | 0.901 |
| Brand diversity@10 (known users) | **0.770** | 0.766 | 0.767 |
| **Explanation fidelity** — cited notes are true (NotePrecision) | 100% | 100% | 100% |
| Recommendations with a concrete reason (Explained@10) | 100% | 100% | 100% |
| Quiz: perfume leaves the top-10 when its **cited** notes are removed | 86.7% | 87.9% | **90.4%** |
| … when as many **uncited** quiz notes are removed (control) | 34.5% | 21.8% | **21.7%** |
| Known users: top-10 perfumes sharing a note with the user's history (Note-grounded@10) | 59.4% | 58.6% | **60.9%** |

## 3. What it says — honestly

1. **Accuracy: notes help, a little.** On known users the note graph is +1.4% NDCG@10 and +1.1% Recall@20 over
   the same model without notes, larger than the seed noise (std 0.0003–0.0004). Same order on validation.
2. **Cold-start NDCG: notes do not win.** With the same training and the same Scorer weights, the quiz path
   scores the same with or without note nodes (0.0487 vs 0.0493 NDCG@10; Recall@20 is higher with notes,
   0.090 vs 0.085). The Week 3 gap (0.052 vs 0.047, `reports/week3_coldstart.json`) came from weights tuned
   per model and from cold models trained with different settings; we report the controlled comparison.
   What makes the quiz work is the **note matching in the Scorer**, which exists only because we built the
   note vocabulary and the perfume–note links — with or without note nodes in the GNN.
3. **Coverage drops with notes: 45% → 39%.** We said it in Week 3 and it is still true. Keeping only the
   3 main notes recovers almost all of it (43.7%, quiz 29%), at no cost on known users — but it costs 18%
   NDCG@10 on the quiz path on *validation* (0.0406 vs 0.0498), and the quiz is our main product path.
   **Decision (validation only): keep all notes** (`configs/default.yaml`, `max_notes_per_item: 0`);
   main notes stay an option for a "discovery" mode. The test numbers confirm the trade-off.
4. **Explanations are faithful, and more so with notes.** Every note we cite is a note of the perfume and of
   the visitor (100%). Removing the cited notes from the quiz pushes ~88% of the explained perfumes out of
   the top-10, against ~22% when we remove as many other quiz notes: the cited notes *are* the reason. Without
   note nodes the control rises to 34.5%: rankings depend less specifically on the notes we name.
5. **So the answer to "is the innovation real?" stays yes**, even where NDCG does not move: note nodes give a
   small but consistent accuracy gain, and the notes make every recommendation explainable with reasons that
   we can test. The price is catalogue coverage, which we measure and can trade back with `--max-notes` or β.

## 4. Frozen for Halima (integration)

`configs/default.yaml` is unchanged in behaviour: user–perfume–note graph, all notes, note edge weight 0.3,
λ_note = 1e-5, liked low-star reviews, seed 3 → served checkpoint `final_nlgcl_notes_rescued_s3.pt`.
The new option is off by default (`max_notes_per_item: 0`).

## Limits
- One cold-start model per config (seed 3); the Scorer weights were tuned in Week 3 on the all-notes model,
  not re-tuned for the main-notes model — part of its quiz loss may come from that.
- Note-grounded@10 and the deletion test measure the *explanations we give*, not the inner workings of the GNN;
  for known users the ranking is the NLGCL score and the explanation is post-hoc (stated in the app).
