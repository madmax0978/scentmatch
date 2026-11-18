"""ScentMatch front end (Streamlit). Talks only to the FastAPI backend.

Run:  streamlit run app/streamlit_app.py      (API must be running on SCENTMATCH_API)
      or scripts/run_demo.sh, which starts both.
"""
import html
import os
import uuid

import requests
import streamlit as st

API = os.getenv("SCENTMATCH_API", "http://localhost:8000")
DEMO_USERS = {"The oud lover": "AEJI7MSPXQCTPRTWBSQYK4NU7AOQ",
              "The designer classic": "AE2RYBGYXARL2Y7U4TC2YAW3FHYQ"}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

st.set_page_config(page_title="ScentMatch", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

for key, default in (("session_user", f"web-{uuid.uuid4().hex[:8]}"), ("quiz_resp", None),
                     ("user_resp", None), ("user_msg", None), ("feedback", {})):
    st.session_state.setdefault(key, default)

# ------------------------------------------------------------------ style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Jost:wght@300;400;500&display=swap');
:root { --ivory:#F7F4EE; --paper:#FFFFFF; --ink:#141414; --taupe:#8A8178; --gold:#A6865A; --line:#E4DED4; }
html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--ivory) !important; color: var(--ink); }
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp div { font-family: 'Jost', sans-serif; font-weight: 300; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
footer, #MainMenu, [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
.block-container { max-width: 1120px; padding: 0 1.5rem 4rem 1.5rem !important; }

/* top bar + hero */
.sm-top { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line);
          padding:18px 0 14px 0; font-size:11px; letter-spacing:.28em; text-transform:uppercase; color:var(--taupe); }
.sm-hero { text-align:center; padding:64px 0 34px 0; }
.stApp .sm-mark { font-family:'Cormorant Garamond',serif; font-weight:400; font-size:58px; letter-spacing:.32em;
           margin:0; padding-left:.32em; color:var(--ink); line-height:1.1; }
.stApp .sm-tag  { font-family:'Cormorant Garamond',serif; font-style:italic; font-size:22px; color:var(--taupe); margin-top:10px; }
.sm-rule { width:56px; height:1px; background:var(--gold); margin:26px auto 0 auto; }

/* section titles */
.sm-kicker { font-size:11px; letter-spacing:.3em; text-transform:uppercase; color:var(--gold); margin:0 0 6px 0; }
.stApp .sm-h2 { font-family:'Cormorant Garamond',serif; font-weight:400; font-size:36px; margin:0 0 6px 0; color:var(--ink); }
.sm-lead { color:var(--taupe); font-size:15px; margin-bottom:28px; }
.sm-q-num { font-size:11px; letter-spacing:.3em; color:var(--gold); text-transform:uppercase; margin:26px 0 2px 0; }
.stApp .sm-q { font-family:'Cormorant Garamond',serif; font-size:25px; margin:0 0 8px 0; color:var(--ink); }

/* mode switch (segmented control) */
[data-testid="stButtonGroup"] [role="radiogroup"] { gap: 10px; flex-wrap: wrap; }
.stApp button[data-variant="segmented_control"] { background:transparent !important; border:none !important;
   border-bottom:1px solid transparent !important; border-radius:0 !important; box-shadow:none !important; padding:6px 6px !important; margin-right:18px; }
.stApp button[data-variant="segmented_control"] p { color:var(--taupe) !important; text-transform:uppercase; letter-spacing:.26em; font-size:12px !important; }
.stApp button[data-variant="segmented_control"][aria-checked="true"] { border-bottom:1px solid var(--ink) !important; }
.stApp button[data-variant="segmented_control"][aria-checked="true"] p { color:var(--ink) !important; }

/* quiz pills */
.stApp button[data-variant="pills"] { border-radius:0 !important; border:1px solid var(--line) !important; background:var(--paper) !important;
   padding:10px 22px !important; transition:all .2s; margin:0 4px 4px 0; }
.stApp button[data-variant="pills"] p { color:var(--ink) !important; font-size:14px !important; letter-spacing:.04em; }
.stApp button[data-variant="pills"]:hover { border-color:var(--ink) !important; }
.stApp button[data-variant="pills"][aria-checked="true"] { background:var(--ink) !important; border-color:var(--ink) !important; }
.stApp button[data-variant="pills"][aria-checked="true"] p { color:var(--ivory) !important; }
[data-testid="stSkillsNudgeAnchor"], [data-testid="stSkillsNudge"] { display:none !important; }

/* buttons */
[data-testid^="stBaseButton-primary"] { background:var(--ink) !important; color:var(--ivory) !important; border:1px solid var(--ink) !important;
   border-radius:0 !important; text-transform:uppercase; letter-spacing:.3em; font-size:12px !important; padding:16px 0 !important; }
[data-testid^="stBaseButton-primary"] p { color:var(--ivory) !important; font-size:12px !important; letter-spacing:.3em; }
[data-testid^="stBaseButton-primary"]:hover { background:var(--gold) !important; border-color:var(--gold) !important; }
[data-testid^="stBaseButton-secondary"] { background:transparent !important; border:1px solid var(--line) !important; border-radius:0 !important;
   color:var(--ink) !important; font-size:11px !important; letter-spacing:.2em; text-transform:uppercase; }
[data-testid^="stBaseButton-secondary"] p { font-size:11px !important; letter-spacing:.2em; }
[data-testid^="stBaseButton-secondary"]:hover { border-color:var(--ink) !important; }

/* inputs */
[data-testid="stForm"] { border:none !important; padding:0 !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div { border-radius:0 !important; }
.stTextInput input, [data-baseweb="select"] > div { background:var(--paper) !important; border-radius:0 !important; border-color:var(--line) !important; }
[data-baseweb="tag"] { background:var(--ink) !important; border-radius:0 !important; }
[data-baseweb="tag"] span { color:var(--ivory) !important; }

/* product cards */
.sm-card { background:var(--paper); border:1px solid var(--line); padding:0 0 18px 0; margin-bottom:10px; min-height:520px; }
.sm-img { height:250px; display:flex; align-items:center; justify-content:center; background:#FFFFFF; border-bottom:1px solid var(--line); }
.sm-img img { max-height:210px; max-width:80%; object-fit:contain; mix-blend-mode:multiply; }
.sm-body { padding:18px 22px 0 22px; }
.stApp .sm-rank { font-family:'Cormorant Garamond',serif; font-style:italic; color:var(--gold); font-size:15px; }
.sm-brand { font-size:11px; letter-spacing:.26em; text-transform:uppercase; color:var(--taupe); margin-top:4px; }
.stApp .sm-name { font-family:'Cormorant Garamond',serif; font-size:25px; line-height:1.15; color:var(--ink); margin:6px 0 10px 0; min-height:58px; }
.sm-notes { font-size:13px; color:var(--taupe); line-height:1.6; min-height:42px; }
.stApp .sm-why { font-family:'Cormorant Garamond',serif; font-style:italic; font-size:17px; color:var(--gold); margin-top:12px; }
.sm-stars { font-size:11px; letter-spacing:.2em; color:var(--taupe); margin-top:8px; }
.sm-chips { text-align:left; margin:4px 0 30px 0; }
.sm-chip { display:inline-block; border:1px solid var(--gold); color:var(--gold); font-size:11px; letter-spacing:.18em;
           text-transform:uppercase; padding:5px 12px; margin:0 8px 8px 0; }
.sm-ok { font-size:11px; letter-spacing:.2em; text-transform:uppercase; color:var(--gold); padding-top:10px; }
.stApp .sm-hist { font-family:'Cormorant Garamond',serif; font-style:italic; font-size:19px; color:var(--ink); margin:2px 0 30px 0; }
.sm-foot { border-top:1px solid var(--line); margin-top:60px; padding-top:18px; font-size:11px; letter-spacing:.2em;
           text-transform:uppercase; color:var(--taupe); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }
.stApp .sm-h2 { line-height:1.15; }
@media (max-width: 700px) { .stApp .sm-mark { font-size:30px; letter-spacing:.18em; padding-left:.18em; }
  .stApp .sm-h2 { font-size:30px; } .sm-top span:last-child { display:none; } }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API helpers
def api(method: str, path: str, quiet_404: bool = False, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=10, **kw)
        if quiet_404 and r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"The ScentMatch service is unavailable ({path}): {e}")
        return None


def log(event: str, item_id: str | None = None, payload: dict | None = None):
    api("POST", "/log_interaction", json={"user_id": st.session_state.session_user,
                                          "item_id": item_id, "event": event, "payload": payload})


@st.cache_data(ttl=600)
def load_quiz():
    return api("GET", "/quiz") or []


@st.cache_data(ttl=600)
def load_notes():
    return api("GET", "/notes") or []


def esc(x) -> str:
    return html.escape(str(x or ""))


# ------------------------------------------------------------------ result grid
def pretty(name: str) -> str:
    """Amazon titles are often in capitals ("MAISON ALHAMBRA KISMET ANGEL"): show them in title case."""
    return name.title() if name.isupper() else name


def card_html(p: dict) -> str:
    cited = [n for n in p["notes"] if n in p["explanation"]]
    notes = cited + [n for n in p["notes"] if n not in cited]
    notes_txt = " · ".join(notes[:7]) + (" …" if len(notes) > 7 else "") if notes else "Notes not listed"
    img = f'<img src="{esc(p["image"])}" alt="{esc(p["name"])}"/>' if p.get("image") else ""
    stars = f'<div class="sm-stars">★ {p["rating"]:.1f} on Amazon</div>' if p.get("rating") else ""
    return (f'<div class="sm-card"><div class="sm-img">{img}</div><div class="sm-body">'
            f'<div class="sm-rank">N° {p["rank"]}</div><div class="sm-brand">{esc(p["brand"])}</div>'
            f'<div class="sm-name">{esc(pretty(p["name"]))}</div><div class="sm-notes">{esc(notes_txt)}</div>'
            f'<div class="sm-why">{esc(p["explanation"])}</div>{stars}</div></div>')


def show(resp: dict | None, tab: str):
    """Results live in session_state, so clicking a button (a Streamlit rerun) keeps them on screen."""
    if not resp:
        return
    if resp["mode"] == "mock":
        st.info("Demo mode: the API is serving placeholder recommendations (no trained model loaded).")
    st.markdown('<div class="sm-kicker">Your selection</div>'
                '<div class="sm-h2">Chosen for you</div>', unsafe_allow_html=True)
    if resp.get("seed_notes"):
        chips = "".join(f'<span class="sm-chip">{esc(n)}</span>' for n in resp["seed_notes"])
        st.markdown(f'<div class="sm-chips">{chips}</div>', unsafe_allow_html=True)
    items = resp["items"]
    for row in range(0, len(items), 3):
        cols = st.columns(3, gap="medium")
        for col, p in zip(cols, items[row:row + 3]):
            with col:
                st.markdown(card_html(p), unsafe_allow_html=True)
                key = (tab, p["parent_asin"])
                b1, b2 = st.columns(2)
                if b1.button("♡ Love it", key=f"like-{tab}-{p['parent_asin']}", use_container_width=True):
                    log("like", p["parent_asin"]); st.session_state.feedback[key] = "like"
                if b2.button("Not for me", key=f"nfm-{tab}-{p['parent_asin']}", use_container_width=True):
                    log("not_for_me", p["parent_asin"]); st.session_state.feedback[key] = "nfm"
                done = st.session_state.feedback.get(key)
                if done:
                    msg = "Added to your wishlist" if done == "like" else "Noted. We will show you less like this"
                    st.markdown(f'<div class="sm-ok">{msg}</div>', unsafe_allow_html=True)
                st.write("")


# ------------------------------------------------------------------ page
health = api("GET", "/health")
st.markdown('<div class="sm-top"><span>Maison ScentMatch</span><span>Haute parfumerie · guided by taste</span></div>'
            '<div class="sm-hero"><div class="sm-mark">SCENTMATCH</div>'
            '<div class="sm-tag">The art of finding your signature scent</div><div class="sm-rule"></div></div>',
            unsafe_allow_html=True)

mode = st.segmented_control("mode", ["Discover your scent", "Your collection"], default="Discover your scent",
                            label_visibility="collapsed", key="mode")
st.write("")

if mode != "Your collection":
    st.markdown('<div class="sm-kicker">The olfactory portrait</div><div class="sm-h2">Six questions, one signature</div>'
                '<div class="sm-lead">Answer as many as you like. Each answer becomes notes, and the notes become a selection.</div>',
                unsafe_allow_html=True)
    questions = load_quiz()
    with st.form("quiz", border=False):
        answers = {}
        for n, q in enumerate(questions):
            st.markdown(f'<div class="sm-q-num">{ROMAN[n]} · {esc(q["id"])}</div><div class="sm-q">{esc(q["question"])}</div>',
                        unsafe_allow_html=True)
            answers[q["id"]] = st.pills(q["question"], q["options"], key=f"q-{q['id']}", label_visibility="collapsed")
        st.markdown('<div class="sm-q-num">VII · notes</div><div class="sm-q">Notes you already adore <span style="font-size:16px;color:#8A8178">(optional)</span></div>',
                    unsafe_allow_html=True)
        extra = st.multiselect("notes", load_notes(), placeholder="vanilla, rose, oud…", label_visibility="collapsed")
        st.write("")
        go = st.form_submit_button("Reveal my selection", type="primary", use_container_width=True)
    if go:
        answers = {k: v for k, v in answers.items() if v}
        if not answers and not extra:
            st.session_state.quiz_resp = None
            st.warning("Choose at least one answer or one note.")
        else:
            log("quiz", payload={"answers": answers, "extra_notes": extra})
            st.session_state.quiz_resp = api("POST", "/recommendations/quiz", json={
                "user_id": st.session_state.session_user, "answers": answers, "extra_notes": extra})
    st.write("")
    show(st.session_state.quiz_resp, "quiz")

else:
    st.markdown('<div class="sm-kicker">Returning client</div><div class="sm-h2">Your collection, continued</div>'
                '<div class="sm-lead">Enter your Amazon reviewer id. The NLGCL model reads the perfumes you loved '
                'and proposes what comes next.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    uid = c1.text_input("Amazon user id", placeholder="e.g. AEJI7MSPXQCTPRTWBSQYK4NU7AOQ", label_visibility="collapsed")
    ask = c2.button("Recommend", type="primary", use_container_width=True)
    demo = st.pills("Try a demo profile", list(DEMO_USERS), key="demo", label_visibility="visible")
    target = uid.strip() if ask and uid.strip() else (DEMO_USERS[demo] if demo and not ask else None)
    if target and target != (st.session_state.user_resp or {}).get("user_id"):
        st.session_state.user_resp = api("GET", f"/recommendations/{target}", quiet_404=True)
        st.session_state.user_msg = None if st.session_state.user_resp else (
            f"We do not know the id “{target}”. Take the olfactory portrait in “Discover your scent” instead.")
    if st.session_state.user_msg:
        st.info(st.session_state.user_msg)
    resp = st.session_state.user_resp
    if resp and resp.get("history"):
        st.markdown('<div class="sm-kicker" style="margin-top:24px">Perfumes you loved</div>'
                    f'<div class="sm-hist">{esc(" · ".join(pretty(h) for h in resp["history"][:8]))}</div>', unsafe_allow_html=True)
    show(resp, "user")

if health:
    st.markdown(f'<div class="sm-foot"><span>ScentMatch · Group 4 · NLGCL graph recommender</span>'
                f'<span>{health["n_perfumes"]:,} perfumes · model: {esc(health["mode"])} · '
                f'{health["n_logged_interactions"]} interactions logged</span></div>', unsafe_allow_html=True)
