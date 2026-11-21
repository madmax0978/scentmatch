"""Week 3 data fixes: note extraction audit, product type, review-text feedback rules."""
import pandas as pd

from scentmatch.data.feedback import label_review
from scentmatch.data.pipeline import extract_notes, product_type


def vocab():
    forms = {"orange": "orange", "orange blossom": "orange blossom", "rose": "rose", "amber": "amber",
             "coconut": "coconut", "vanilla": "vanilla"}
    return {"surface_forms": forms}


def notes_of(text):
    df = extract_notes(pd.DataFrame({"parent_asin": ["A"], "text": [text]}), vocab())
    return set(df["note"])


def test_plural_notes_are_not_cut_short():
    assert notes_of("Top notes: orange blossoms and vanilla") == {"orange blossom", "vanilla"}


def test_non_scent_uses_are_ignored():
    assert notes_of("rose gold cap, amber glass bottle, enriched with coconut oil") == set()
    assert notes_of("notes of rose, amber and coconut") == {"rose", "amber", "coconut"}


def test_product_type():
    assert product_type("Versace Eros 3.4 oz EDT Spray") == "perfume"
    assert product_type("Dolce & Gabbana The One 2-Pieces Set For Men") == "set"
    assert product_type("Ariana Grande Ari Body Mist 8 oz") == "body"
    assert product_type("Tom Ford Noir 10ml rollerball") == "mini"


def test_review_rules():
    assert label_review(5, "meh")[0] == "like"                                     # >= 4 stars unchanged
    assert label_review(3, "Smells amazing but it fades after an hour")[0] == "like"
    assert label_review(1, "Smells like an old lady, gave me a headache")[0] == "dislike"
    assert label_review(1, "Bottle arrived broken and leaked everywhere")[0] == "not_scent"
    assert label_review(1, "This is a fake, nothing like the original")[0] == "not_scent"
    assert label_review(2, "Doesn't last more than 20 minutes")[0] == "not_scent"
    assert label_review(2, "It doesn't smell good at all")[0] == "dislike"
    assert label_review(3, "ok")[0] == "unclear"
