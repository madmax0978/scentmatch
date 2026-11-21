"""Explicit feedback from review texts (Data Engineer, Week 3).

Professor's feedback: dropping the low Amazon ratings biases everything. Until Week 2, only
4-5 star ratings were used ("liked") and every 1-3 star rating was thrown away. But a star
rating mixes several things: for a perfume, 1 star can mean "I hate this smell" (a real
taste signal) or "the bottle arrived broken", "it is a fake", "it does not last" (no
information about taste at all). And a 3-star review is often "smells amazing, but it fades
after an hour" - a *liked* scent.

So we read the text of every 1-3 star review of our users and decide, with explainable
keyword rules (no black box), what it says about the scent:

    label = "dislike"    the text criticises the smell itself          -> hard negative
            "like"       the text praises the smell, the low score is  -> extra positive
                         about something else (longevity, price...)
            "not_scent"  only about delivery, packaging, sprayer,      -> ignored
                         authenticity or longevity
            "unclear"    no reason found: 1-2 stars -> "dislike" by default (the most likely
                         reading), 3 stars -> ignored

Each review also gets `period`: "train" if it was written before the user's validation
interaction, else "eval". Only "train" feedback may be used by the model (no leakage);
"eval" dislikes are used to *evaluate* whether the model can tell likes from dislikes.

Input : data/raw/kcore_reviews.parquet (scentmatch.data.reviews)
Output: data/processed/feedback.parquet, reports/audits/feedback_stats.json
Run   : python -m scentmatch.data.feedback
"""
from __future__ import annotations

import json
import re

import pandas as pd

from scentmatch.config import load_config

SMELL = r"(smell|smells|smelled|smelling|scent|scents|fragrance|perfume|cologne|aroma|notes?)"
GOOD = (r"(good|great|amazing|wonderful|lovely|beautiful|nice|fantastic|awesome|divine|delicious|"
        r"heavenly|incredible|gorgeous|pleasant|fabulous|excellent|sexy|yummy|intoxicating|"
        r"fresh and clean|clean and fresh|like heaven)")
BAD = (r"(bad|awful|terrible|horrible|disgusting|gross|nasty|cheap|chemical|harsh|sour|rancid|"
       r"vile|unpleasant|sickening|overpowering|nauseating|weird|strange|old lady|off|musty|"
       r"like alcohol|like bug spray|like rubbing alcohol|like plastic|like cat pee|like urine|"
       r"like soap|like a candle|like baby powder|like an old)")
NEG = r"(not|n't|never|no longer|hardly|barely)"

RULES: dict[str, list[str]] = {
    # the smell itself is liked
    "scent_pos": [
        rf"\b{SMELL}\s+(is|was|are|were)?\s*(so|very|really|super|pretty|absolutely|just)?\s*{GOOD}",
        rf"\b{GOOD}\s+(\w+\s+)?{SMELL}\b",
        rf"\b(smells?|scent)\s+(\w+\s+){{0,4}}(wonderful|amazing|great|lovely|divine|fantastic|heavenly)\b",
        r"\b(love|loved|loves|like|liked|adore|enjoy|enjoyed)\s+(the|this|its|how it)\s+(smell|scent|fragrance|smells)",
        r"\bsmells?\s+(so\s+|really\s+|very\s+)?(good|great|amazing|nice|wonderful|lovely|divine|delicious|heavenly)",
        r"\b(compliments?|complimented)\b",
        r"\blove (this|it)\b(?!.{0,15}\bbut\b.{0,30}\b(smell|scent))",
        r"\b(me encanta|buen (aroma|olor)|olor agradable|huele (rico|bien|delicioso))\b",
    ],
    # the smell itself is disliked
    "scent_neg": [
        rf"\b{SMELL}\s+(is|was|are|were)?\s*(so|very|really|too|just|kind of|kinda)?\s*{BAD}",
        rf"\bsmells?\s+(so\s+|really\s+|very\s+|kind of\s+|kinda\s+)?{BAD}",
        rf"\b{BAD}\s+(smell|scent|fragrance)\b",
        rf"\b(do|does|did)\s?{NEG}\s+(like|love|care for|enjoy)\s+(the|this|its|how it)?\s*(smell|scent|fragrance|it)",
        r"\b(hate|hated|dislike|disliked|can'?t stand|couldn'?t stand)\s+(the|this|its)?\s*(smell|scent|fragrance|it)",
        r"\b(headache|migraine|nause|gag|gagged|sneez|made me sick|makes me sick)",
        r"\bnot (my|for me|a fan)|\bnot my (type|style|cup of tea|scent)",
        r"\btoo (sweet|strong|powdery|musky|floral|fruity|perfumey|heavy|manly|feminine|masculine|young|old|spicy)\b",
        r"\b(stinks?|stunk|stinky|reeks)\b",
        r"\b(does|did|do)\s?(n'?t|not)\s+smell\s+(good|nice|great|pleasant|right)",
        r"\bsmells?\s+(\w+\s+)?like\s+(a\s+|an\s+)?(cheap|plastic|rubber|old|chemical|alcohol|bug|insect|cleaning|"
        r"cleaner|disinfectant|lysol|mothball|grandma|medicine|band-?aid|cigarette|ashtray|vomit|sweat)",
        r"\b(chemical|alcohol|antiseptic|medicinal|synthetic)\s+(smell|smelling|scent|undertone)",
        r"\b(mature|old (woman|lady|man|people)|grandma|granny)\s*(smell|smelling|scent)",
        r"\bnot (a|the) (huge |big |real )?fan\b|\bnot it for me\b|\bnot for me\b",
        r"\b(can'?t|couldn'?t|could not|cannot)\s+(come to\s+)?(like|stand|tolerate|wear)\b",
        r"\bno me (gusta|gusto|gustó|agrada|agradó)\b|\bhuele (mal|horrible|feo)\b",
        rf"\b{NEG}\s+(a\s+)?(good|nice|pleasant|great)\s+(smell|scent)",
        rf"\b(smell|scent)\s+(is|was)?\s*{NEG}\s+(good|nice|pleasant|great)",
    ],
    # the scent does not last (says nothing about liking the smell)
    "longevity": [
        rf"\b(does|did|do)\s?{NEG}\s+(last|stay|linger)",
        r"\b(fades?|faded|fading|wears? off|wore off|disappears?|disappeared|evaporates?|vanish)",
        r"\b(no|poor|zero|bad|little) (longevity|staying power|sillage|projection)",
        r"\b(weak|faint|light)\b.{0,20}\b(smell|scent|fragrance)|\b(barely|can'?t|cannot|could not|couldn'?t) (smell|even smell)",
        r"\b(watered down|diluted|weak)\b",
        r"\blasts?\s+(for\s+)?(only\s+)?(a|one|about|maybe|like)?\s*(minute|few minutes|second|hour|\d+\s*(min|minutes|hours?))",
        r"\b(short[- ]lived|not lasting|lasting power|staying power|smells only for)",
    ],
    # delivery, packaging, sprayer, wrong item
    "product": [
        r"\b(leak|leaked|leaking|leaks|broken|broke|cracked|shattered|damaged|spilled)\b",
        r"\b(arrived|arrive|shipping|shipped|delivery|delivered|package|packaging|box was|seal)\b",
        r"\b(sprayer|nozzle|atomizer|pump|spray(er)? (does|did)n'?t work|won'?t spray|doesn'?t spray|didn'?t spray)",
        r"\b(empty|half (full|empty)|missing|wrong (item|product|one|size)|not what i ordered|expired)\b",
        r"\b(return|returned|refund)\b",
    ],
    # counterfeit / reformulation: a judgement on the copy, not on the perfume's scent
    "authenticity": [
        r"\b(fake|counterfeit|knock ?off|not (the )?(real|authentic|genuine|original)|imitation|bootleg)\b",
        r"\b(smells?|smelled) (nothing|different|not) like (the )?(original|real|one i|what i|the one)",
        r"\b(reformulat|not the same as|different from (the )?(one|store|original))",
        r"\b(imposter|impostor)\b|(n't|not|nothing)\s+(\w+\s+){0,3}smells?\s+like\s+(the\s+)?(original|real)",
    ],
    "price": [r"\b(overpriced|too expensive|not worth|waste of money|price)\b"],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in RULES.items()}


_NEGATED = re.compile(r"\b(not|never|no|nothing|n't)\b|n't\b", re.I)


def _found(family: str, pats: list[re.Pattern], text: str) -> bool:
    for p in pats:
        for m in p.finditer(text):
            if family == "scent_neg":
                # "not too strong", "never too sweet" are not complaints
                if re.search(r"\b(not|never|n't)\s*$", text[max(0, m.start() - 8):m.start()], re.I):
                    continue
                return True
            if family != "scent_pos":
                return True
            # negation guard: "doesn't smell good", "not a nice scent" are not praise
            if not _NEGATED.search(text[max(0, m.start() - 12):m.end()]):
                return True
    return False


def reasons(text: str) -> set[str]:
    """Which reason families appear in the review (title + text)."""
    return {k for k, pats in _COMPILED.items() if _found(k, pats, text)}


def label_review(rating: float, text: str) -> tuple[str, list[str]]:
    """Return (label, reasons) for one review. Ratings >= 4 stay 'like' (unchanged split)."""
    r = reasons(text or "")
    if rating >= 4:
        return "like", sorted(r)
    if "scent_neg" in r and not ("authenticity" in r and "scent_pos" in r):
        if "authenticity" in r:                     # "smells awful, nothing like the original"
            return "not_scent", sorted(r)
        return "dislike", sorted(r)
    if "scent_pos" in r:
        return "like", sorted(r)                    # liked smell, complaint about something else
    if r & {"product", "authenticity", "longevity", "price"}:
        return "not_scent", sorted(r)
    return ("dislike" if rating <= 2 else "unclear"), sorted(r)


def build_feedback(cfg: dict) -> tuple[pd.DataFrame, dict]:
    proc = cfg["paths"]["processed_dir"]
    reviews = pd.read_parquet(cfg["paths"]["raw_dir"] / "kcore_reviews.parquet")
    users = pd.read_parquet(proc / "users.parquet")
    items = pd.read_parquet(proc / "items.parquet", columns=["parent_asin", "item_idx"])
    val = pd.read_parquet(proc / "val.parquet")[["user_idx", "timestamp"]].rename(columns={"timestamp": "val_ts"})

    df = (reviews.merge(users, on="user_id").merge(items, on="parent_asin")
                 .merge(val, on="user_idx", how="left"))
    text = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.replace("<br />", " ", regex=False)
    labels = [label_review(r, t) for r, t in zip(df["rating"], text)]
    df["label"] = [lab for lab, _ in labels]
    df["reasons"] = [",".join(rs) for _, rs in labels]
    df["period"] = (df["timestamp"] < df["val_ts"]).map({True: "train", False: "eval"})

    # The rating file (scentmatch.data.pipeline) is the source of truth: when a user reviewed the
    # same perfume twice (e.g. 2 stars, later 5 stars), the pair is a positive there; drop it here.
    positives = pd.concat([pd.read_parquet(proc / f"{s}.parquet", columns=["user_idx", "item_idx"])
                           for s in ("train", "val", "test")])
    pos_keys = set(zip(positives["user_idx"], positives["item_idx"]))
    is_pos = pd.Series([k in pos_keys for k in zip(df["user_idx"], df["item_idx"])], index=df.index)
    low = df[(df["rating"] <= 3) & ~is_pos]
    stats = {
        "low_star_pairs_dropped_because_positive_in_rating_file": int(((df["rating"] <= 3) & is_pos).sum()),
        "reviews_found": len(df),
        "ratings_1_3": len(low),
        "label_by_stars": {str(int(s)): g["label"].value_counts().to_dict() for s, g in low.groupby("rating")},
        "reasons_in_1_3_star": {k: int(low["reasons"].str.contains(k).sum()) for k in RULES},
        "positives_with_scent_complaint": int(((df["rating"] >= 4) & df["reasons"].str.contains("scent_neg")
                                               & ~df["reasons"].str.contains("scent_pos")).sum()),
        "train_period": {"dislike": int(((low["label"] == "dislike") & (low["period"] == "train")).sum()),
                         "like": int(((low["label"] == "like") & (low["period"] == "train")).sum())},
        "eval_period": {"dislike": int(((low["label"] == "dislike") & (low["period"] == "eval")).sum())},
        "users_with_train_dislike": int(low[(low["label"] == "dislike") & (low["period"] == "train")]["user_idx"].nunique()),
    }
    out = low[["user_idx", "item_idx", "timestamp", "rating", "label", "reasons", "period"]].reset_index(drop=True)
    return out, stats, df


def main() -> None:
    cfg = load_config()
    out, stats, df = build_feedback(cfg)
    proc = cfg["paths"]["processed_dir"]
    out.to_parquet(proc / "feedback.parquet", index=False)
    audit_dir = proc.parents[1] / "reports" / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "feedback_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
