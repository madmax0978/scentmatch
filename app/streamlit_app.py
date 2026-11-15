"""ScentMatch front-end (Streamlit). Talks only to the FastAPI backend.

Run:  streamlit run app/streamlit_app.py      (API must be running on SCENTMATCH_API)
"""
import os
import uuid

import requests
import streamlit as st

API = os.getenv("SCENTMATCH_API", "http://localhost:8000")
st.set_page_config(page_title="ScentMatch", page_icon="🌸", layout="wide")

if "session_user" not in st.session_state:
    st.session_state.session_user = f"web-{uuid.uuid4().hex[:8]}"


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=5, **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error on {path}: {e}")
        return None


def log(event: str, item_id: str | None = None, payload: dict | None = None):
    api("POST", "/log_interaction", json={"user_id": st.session_state.session_user,
                                          "item_id": item_id, "event": event, "payload": payload})


def show(resp):
    if not resp:
        return
    if resp["mode"] == "mock":
        st.info("Serving mode: **mock** (Week 2) - recommendations are placeholders from the real catalogue.")
    if resp.get("seed_notes"):
        st.caption("Seed notes: " + ", ".join(resp["seed_notes"]))
    for p in resp["items"]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 5, 2])
            if p["image"]:
                c1.image(p["image"], width=80)
            c2.markdown(f"**{p['rank']}. {p['title']}**  \n{p['brand']}")
            c2.caption(", ".join(p["notes"][:8]) or "no notes extracted")
            c2.write(f"_{p['explanation']}_")
            if c3.button("Like", key=f"like-{p['parent_asin']}"):
                log("like", p["parent_asin"]); c3.success("Saved")
            if c3.button("Not for me", key=f"nfm-{p['parent_asin']}"):
                log("not_for_me", p["parent_asin"]); c3.warning("Noted")


st.title("ScentMatch")
st.write("Perfume recommendations from the notes you love.")
health = api("GET", "/health")
if health:
    st.sidebar.write(f"API: `{API}`  \nmode: **{health['mode']}**  \nperfumes: {health['n_perfumes']:,}  \n"
                     f"logged events: {health['n_logged_interactions']}")

tab_quiz, tab_user = st.tabs(["New here? Take the quiz", "I already rated perfumes"])

with tab_quiz:
    questions = api("GET", "/quiz") or []
    with st.form("quiz"):
        answers = {q["id"]: st.radio(q["question"], q["options"], horizontal=True) for q in questions}
        extra = st.text_input("Notes you already know you love (optional, comma separated)")
        go = st.form_submit_button("Find my perfumes")
    if go:
        extra_notes = [x.strip().lower() for x in extra.split(",") if x.strip()]
        log("quiz", payload={"answers": answers, "extra_notes": extra_notes})
        show(api("POST", "/recommendations/quiz", json={"user_id": st.session_state.session_user,
                                                        "answers": answers, "extra_notes": extra_notes}))

with tab_user:
    uid = st.text_input("Amazon user id", placeholder="e.g. AE2...")
    if st.button("Recommend") and uid:
        show(api("GET", f"/recommendations/{uid}"))
