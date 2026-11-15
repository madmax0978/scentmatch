"""Taste quiz -> seed notes (fixed, documented rule table).

Questions ask about moments, tastes and places that map directly to olfactory notes,
not about personality (Janci & Sotakova, 2022 found only a weak link between Big Five
traits and scent preference). Every note below must exist in data/processed/notes.json
(checked by tests/test_api.py).
"""
from __future__ import annotations

QUIZ: list[dict] = [
    {"id": "moment", "question": "When would you wear it most?",
     "options": {"Morning walk": ["bergamot", "lemon", "neroli"],
                 "At the office": ["iris", "musk", "lavender"],
                 "Evening out": ["amber", "oud", "vanilla"],
                 "Date night": ["rose", "patchouli", "tonka bean"]}},
    {"id": "taste", "question": "Sweet, fresh, spicy or earthy?",
     "options": {"Sweet": ["vanilla", "tonka bean", "caramel"],
                 "Fresh": ["mandarin orange", "grapefruit", "sea notes"],
                 "Spicy": ["cardamom", "pink pepper", "cinnamon"],
                 "Earthy": ["vetiver", "patchouli", "oakmoss"]}},
    {"id": "season", "question": "Your favourite season?",
     "options": {"Spring": ["peony", "lily of the valley", "jasmine"],
                 "Summer": ["citrus accord", "sea notes", "coconut"],
                 "Autumn": ["sandalwood", "cedar", "tobacco"],
                 "Winter": ["amber", "leather", "frankincense"]}},
    {"id": "treat", "question": "Pick a treat",
     "options": {"Coffee": ["coffee", "caramel"],
                 "Earl Grey tea": ["bergamot", "tea"],
                 "Fresh fruit": ["peach", "apple", "blackcurrant"],
                 "Dark chocolate": ["cocoa", "tonka bean"]}},
    {"id": "place", "question": "Where do you feel best?",
     "options": {"A rose garden": ["rose", "jasmine", "gardenia"],
                 "A forest": ["cedar", "vetiver", "oakmoss"],
                 "The beach": ["sea notes", "coconut", "orange blossom"],
                 "An old library": ["leather", "tobacco", "iris"]}},
    {"id": "intensity", "question": "Soft skin scent or bold trail?",
     "options": {"Soft and close to the skin": ["musk", "white musk", "powdery notes"],
                 "Somewhere in between": ["sandalwood", "jasmine"],
                 "Bold, noticed from afar": ["oud", "amber", "leather"]}},
]


def seed_notes(answers: dict[str, str], extra_notes: list[str] | None = None) -> list[str]:
    """Map quiz answers {question_id: option} to an ordered, de-duplicated note list."""
    notes: list[str] = []
    by_id = {q["id"]: q for q in QUIZ}
    for qid, option in answers.items():
        if qid not in by_id:
            raise KeyError(f"unknown question '{qid}'")
        if option not in by_id[qid]["options"]:
            raise KeyError(f"unknown option '{option}' for question '{qid}'")
        notes.extend(by_id[qid]["options"][option])
    notes.extend(extra_notes or [])
    return list(dict.fromkeys(n.lower() for n in notes))


def all_quiz_notes() -> set[str]:
    return {n for q in QUIZ for opts in q["options"].values() for n in opts}
