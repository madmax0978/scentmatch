"""Olfactory note vocabulary.

Built from the open Parfumo dataset (TidyTuesday 2024-12-10, 59,325 perfumes with
top / middle / base notes). Origin qualifiers are stripped ("Italian bergamot" ->
"bergamot") and a hand-written synonym table adds spellings found in Amazon text
("vanille" -> "vanilla", "agarwood" -> "oud").
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

NOTE_COLUMNS = ["Top_Notes", "Middle_Notes", "Base_Notes"]

# Geographic / grade qualifiers that do not change the olfactory note.
QUALIFIERS = {
    "italian", "calabrian", "sicilian", "sicilian", "bulgarian", "turkish", "egyptian",
    "moroccan", "indian", "madagascan", "madagascar", "bourbon", "tahitian", "mexican",
    "haitian", "indonesian", "virginia", "virginian", "atlas", "texas", "mysore",
    "australian", "french", "spanish", "tunisian", "persian", "chinese", "japanese",
    "cambodian", "laotian", "brazilian", "peruvian", "somalian", "omani", "guatemalan",
    "javanese", "sri", "lankan", "ceylon", "grasse", "damask", "damascena", "centifolia",
    "may", "absolute", "essence", "oil", "extract", "accord", "leaf",
    "co2", "natural", "synthetic", "tunisian", "florentine", "vietnamese", "laos",
    "lebanon", "haitian", "provence", "provencal", "english", "russian", "nepalese",
}

# Spellings seen in Amazon titles/descriptions -> canonical vocabulary entry.
SYNONYMS = {
    "vanille": "vanilla", "vanila": "vanilla", "bourbon vanilla": "vanilla",
    "agarwood": "oud", "oudh": "oud", "aoud": "oud", "oud wood": "oud",
    "tonka": "tonka bean", "tonka beans": "tonka bean", "feve tonka": "tonka bean",
    "cedarwood": "cedar", "cedar wood": "cedar", "sandal wood": "sandalwood",
    "santal": "sandalwood", "musks": "musk", "white musks": "white musk",
    "patchouly": "patchouli", "jasmin": "jasmine", "sambac": "jasmine",
    "ylang ylang": "ylang-ylang", "ylang": "ylang-ylang", "lily-of-the-valley": "lily of the valley",
    "muguet": "lily of the valley", "orange flower": "orange blossom", "neroli oil": "neroli",
    "mandarin": "mandarin orange", "tangerine": "mandarin orange", "bergamote": "bergamot",
    "vetyver": "vetiver", "oak moss": "oakmoss", "olibanum": "frankincense", "incense": "frankincense",
    "black currant": "blackcurrant", "cassis": "blackcurrant", "pink peppercorn": "pink pepper",
    "tuberose": "tuberose", "gardenia": "gardenia", "heliotrope": "heliotrope",
    "caramel": "caramel", "praline": "praline", "coffee": "coffee", "cacao": "cocoa",
    "chocolate": "cocoa", "sea salt": "sea notes", "marine": "sea notes", "ocean": "sea notes",
    "aquatic": "sea notes", "rosewood": "rosewood", "rose de mai": "rose", "roses": "rose",
    "violets": "violet", "lilies": "lily", "peonies": "peony", "citrus": "citruses",
    "citrusy": "citruses", "leathery": "leather", "smoky": "smoke", "tobacco leaf": "tobacco",
    "honeyed": "honey", "powdery": "powdery notes", "woody": "woody notes",
}

# Words that are vocabulary notes but too ambiguous in product text.
AMBIGUOUS = {"water", "air", "skin", "salt", "sugar", "milk", "cream", "wood", "woods",
             "flowers", "fruits", "spices", "green notes", "fresh", "clean", "soap",
             "paper", "metal", "glass", "silk", "cotton", "nausea", "ozone", "earth",
             "sweet", "ice", "salty", "sea", "bitter", "warm", "cool", "light", "dark",
             "gold", "black", "red", "white", "blue", "green", "pink", "love", "night",
             # Week 3 audit: "orange leaves", "violet leaves" -> the plant is the note, not "leaves";
             # "powder" is mostly a product ("dusting powder"), the note is "powdery notes"
             "leaves", "powder", "butter"}

# Week 3 audit: a note word followed by one of these words is not a scent note in the text
# ("rose gold accents", "amber glass bottle", "enriched with coconut oil").
NOT_A_NOTE_AFTER = {
    "rose": r"gold",
    "amber": r"glass|bottle|colou?r|colou?red|vial",
    "coconut": r"oil|butter|milk lotion",
    "almond": r"oil|butter",
    "aloe": r"vera|mist|gel|juice",
    "honey": r"comb applicator",
    "cocoa": r"butter|orchid",          # "cacao butter" (skin care), "chocolate orchid" (a flower)
}


ACCORD_EXCLUDE = {"synthetic", "fresh", "sweet", "green", "creamy"}
ACCORD_ALIASES = {"floriental": "oriental", "citrusy": "citrus", "woodsy": "woody",
                  "fruit": "fruity", "flowery": "floral", "spice": "spicy"}


def canonical(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip().lower())
    name = SYNONYMS.get(name, name)
    words = [w for w in name.split(" ") if w not in QUALIFIERS]
    name = " ".join(words) if words else name
    return SYNONYMS.get(name, name)


def build_vocabulary(parfumo_csv: Path, min_freq: int = 20) -> dict:
    df = pd.read_csv(parfumo_csv, usecols=NOTE_COLUMNS + ["Main_Accords"])
    counts: Counter = Counter()
    for col in NOTE_COLUMNS:
        for cell in df[col].dropna():
            notes = {canonical(x) for x in cell.split(",") if x.strip()}
            counts.update(notes)
    notes = sorted(n for n, c in counts.items()
                   if c >= min_freq and len(n) >= 3 and n not in AMBIGUOUS)
    note_set = set(notes)
    # surface form -> canonical note; only keep synonyms whose target is in the vocabulary
    surface = {n: n for n in notes}
    for alias, target in SYNONYMS.items():
        target = canonical(target)
        if target in note_set:
            surface[alias] = target

    # Accords (fallback from the Week 1 risk table): "floral" -> "floral accord".
    accord_counts: Counter = Counter(
        a.strip().lower() for cell in df["Main_Accords"].dropna() for a in cell.split(","))
    accords = sorted(a for a, c in accord_counts.items()
                     if c >= min_freq and a not in AMBIGUOUS and a not in ACCORD_EXCLUDE)
    for a in accords:
        surface.setdefault(a, f"{a} accord")
    surface.update({alias: f"{a} accord" for alias, a in ACCORD_ALIASES.items() if a in accords})
    return {
        "source": "Parfumo (TidyTuesday 2024-12-10), notes and accords in >= %d perfumes" % min_freq,
        "n_notes": len(notes),
        "n_accords": len(accords),
        "notes": notes,
        "accords": [f"{a} accord" for a in accords],
        "counts": {n: counts[n] for n in notes},
        "surface_forms": surface,
    }


def save_vocabulary(vocab: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, indent=1, ensure_ascii=False))


def load_vocabulary(path: Path) -> dict:
    return json.loads(path.read_text())
