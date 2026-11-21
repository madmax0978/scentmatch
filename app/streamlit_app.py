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

# The visitor id lives in the URL (?visitor=web-1a2b3c4d): reloading the page or coming back with
# the same link keeps the quiz, the likes and the rejections (they are read back from SQLite).
if "session_user" not in st.session_state:
    vid = st.query_params.get("visitor", "")
    st.session_state.session_user = vid if vid.startswith("web-") and len(vid) <= 40 else f"web-{uuid.uuid4().hex[:8]}"
    st.query_params["visitor"] = st.session_state.session_user
for key, default in (("quiz_resp", None),
                     ("user_resp", None), ("user_msg", None), ("similar", None)):
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
.sm-why-box { font-size:13px; color:var(--ink); line-height:1.5; }
.sm-bar-row { display:flex; align-items:center; gap:8px; margin:3px 0; font-size:12px; }
.sm-bar-label { width:120px; color:var(--taupe); overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.sm-bar { height:6px; background:var(--gold); }
.sm-bar.neg { background:#B5655A; }
.sm-bar-val { color:var(--taupe); font-size:11px; }
.sm-adapt { border:1px solid var(--gold); padding:12px 18px; margin:6px 0 24px 0; font-size:13px; color:var(--ink); }
.sm-chips { text-align:left; margin:4px 0 30px 0; }
.sm-chip { display:inline-block; border:1px solid var(--gold); color:var(--gold); font-size:11px; letter-spacing:.18em;
           text-transform:uppercase; padding:5px 12px; margin:0 8px 8px 0; }
.sm-ok { font-size:11px; letter-spacing:.2em; text-transform:uppercase; color:var(--gold); padding-top:10px; }
.stApp .sm-hist { font-family:'Cormorant Garamond',serif; font-style:italic; font-size:19px; color:var(--ink); margin:2px 0 30px 0; }
.sm-choice { background:var(--paper); border:1px solid var(--line); padding:12px 14px; margin-bottom:6px; display:flex; gap:12px; align-items:center; height:118px; overflow:hidden; }
.sm-choice img { width:56px; height:56px; object-fit:contain; mix-blend-mode:multiply; }
.sm-choice .nm { font-family:"Cormorant Garamond",serif; font-size:19px; line-height:1.15; color:var(--ink); display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.sm-choice .br { font-size:10px; letter-spacing:.22em; text-transform:uppercase; color:var(--taupe); }
.sm-state { font-size:10px; letter-spacing:.22em; text-transform:uppercase; padding:2px 8px; border:1px solid var(--gold); color:var(--gold); display:inline-block; margin-top:4px; }
.sm-state.no { border-color:#B5655A; color:#B5655A; }
.sm-empty { color:var(--taupe); font-style:italic; font-size:14px; margin:4px 0 20px 0; }
.sm-sep { border-top:1px solid var(--line); margin:46px 0 30px 0; }
.sm-foot { border-top:1px solid var(--line); margin-top:60px; padding-top:18px; font-size:11px; letter-spacing:.2em;
           text-transform:uppercase; color:var(--taupe); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }
.stApp .sm-h2 { line-height:1.15; }
@media (max-width: 700px) { .stApp .sm-mark { font-size:30px; letter-spacing:.18em; padding-left:.18em; }
  .stApp .sm-h2 { font-size:30px; } .sm-top span:last-child { display:none; } }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API helpers
def api(method: str, path: str, quiet_404: bool = False, quiet: tuple[int, ...] = (), **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=10, **kw)
        if (quiet_404 and r.status_code == 404) or r.status_code in quiet:
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
    return (f'<div class="sm-card"><div class="sm-img">{img}</div><div class="sm-body">'
            f'<div class="sm-rank">N° {p["rank"]}</div><div class="sm-brand">{esc(p["brand"])}</div>'
            f'<div class="sm-name">{esc(pretty(p["name"]))}</div><div class="sm-notes">{esc(notes_txt)}</div>'
            f'<div class="sm-why">{esc(p["explanation"])}</div></div></div>')


def bar(label: str, value: float, scale: float) -> str:
    width = int(140 * min(abs(value) / scale, 1.0)) if scale > 0 else 0
    neg = " neg" if value < 0 else ""
    return (f'<div class="sm-bar-row"><span class="sm-bar-label">{esc(label)}</span>'
            f'<span class="sm-bar{neg}" style="width:{max(width, 2)}px"></span>'
            f'<span class="sm-bar-val">{value:+.3f}</span></div>')


def why_html(p: dict) -> str:
    """The parts of the score, as returned by the API (see scentmatch.train.predict)."""
    b = p.get("breakdown") or {}
    if not b:
        return '<div class="sm-why-box">No breakdown in mock mode.</div>'
    rows = []
    parts = [v for v in (b.get("notes_part"), b.get("collab_part"), b.get("avoid_part")) if v is not None]
    scale = max([abs(v) for v in parts] + [1e-6])
    rows.append(bar("Your notes", b.get("notes_part", 0), scale))
    for n in b.get("notes", [])[:4]:
        rows.append(bar(f"   · {n['note']}", n["contribution"], scale))
    rows.append(bar("Taste (NLGCL)", b.get("collab_part", 0), scale))
    for s in b.get("similar_liked", [])[:3]:
        rows.append(bar(f"   · {pretty(s['name'])}", s["contribution"], scale))
    if b.get("avoid_part"):
        rows.append(bar("Turned down", b["avoid_part"], scale))
    head = (f'Score {b.get("total", 0):.3f} = {b.get("w_notes", 0):.2f} × notes + {b.get("w_collab", 0):.2f} × taste'
            f'{" − rejections" if b.get("avoid_part") else ""}. Popularity weight β = {b.get("beta", 0)}.')
    if b.get("explanation_type"):
        head = f'NLGCL score {b.get("model_score", 0):.2f}, explained by your closest perfumes and notes.'
    return f'<div class="sm-why-box">{esc(head)}{"".join(rows)}</div>'


def uid_of(tab: str) -> str:
    """The quiz tab is the anonymous visitor; the collection tab is the Amazon profile on screen."""
    return st.session_state.session_user if tab == "quiz" else st.session_state.user_resp["user_id"]


def refresh(tab: str):
    """Recompute the selection from everything the visitor did (quiz + likes + rejections)."""
    uid = uid_of(tab)
    if tab == "quiz":
        # 422 = nothing left to build on (no quiz, every reaction undone): clear the grid quietly
        st.session_state.quiz_resp = api("POST", "/recommendations/session", quiet=(422,),
                                         json={"user_id": uid, "k": 9})
        return
    history = (st.session_state.user_resp or {}).get("history")
    state = api("GET", f"/session/{uid}") or {}
    if state.get("reactions"):
        resp = api("POST", "/recommendations/session", json={"user_id": uid, "k": 9})
    else:                                   # back to the pure NLGCL ranking of this client
        resp = api("GET", f"/recommendations/{uid}?k=9", quiet_404=True)
    if resp:
        resp["history"] = history
        resp["reactions"] = resp.get("reactions") or []
        st.session_state.user_resp = resp


def react(tab: str, asin: str, event: str):
    """Button callback. event = like | not_for_me | undo. Clicking the state a perfume is
    already in undoes it, so every choice can be taken back or switched."""
    current = reactions_of(tab).get(asin)
    if event == current:
        event = "undo"
    api("POST", "/log_interaction", json={"user_id": uid_of(tab), "item_id": asin, "event": event})
    refresh(tab)


def clear_all(tab: str, asins: list[str]):
    for a in asins:
        api("POST", "/log_interaction", json={"user_id": uid_of(tab), "item_id": a, "event": "undo"})
    refresh(tab)


def reactions_of(tab: str) -> dict[str, str]:
    """{asin: like | not_for_me} as last returned by the API for this tab."""
    resp = st.session_state.quiz_resp if tab == "quiz" else st.session_state.user_resp
    if tab == "quiz" and not resp:          # no selection on screen: read the log directly
        resp = api("GET", f"/session/{st.session_state.session_user}") or {}
    return {r["parent_asin"]: r["reaction"] for r in (resp or {}).get("reactions") or []}


def reaction_buttons(tab: str, asin: str, key: str, current: str | None):
    b1, b2 = st.columns(2)
    b1.button("♡ Loved · undo" if current == "like" else "♡ Love it", key=f"like-{key}", use_container_width=True,
              on_click=react, args=(tab, asin, "like"))
    b2.button("✕ Undo" if current == "not_for_me" else "✕ Not for me", key=f"nfm-{key}", use_container_width=True,
              on_click=react, args=(tab, asin, "not_for_me"))


def choices_panel(reactions: list[dict], tab: str, title: str, empty: str):
    """Every perfume the visitor loved or turned down, each one editable: undo, or switch side."""
    st.markdown(f'<div class="sm-kicker">{esc(title)}</div>', unsafe_allow_html=True)
    if not reactions:
        st.markdown(f'<div class="sm-empty">{esc(empty)}</div>', unsafe_allow_html=True)
        return
    for kind, label in (("like", "♡ Loved"), ("not_for_me", "✕ Not for me")):
        group = [r for r in reactions if r["reaction"] == kind]
        st.markdown(f'<div class="sm-q" style="font-size:21px;margin-top:10px">{label} '
                    f'<span style="font-size:15px;color:#8A8178">({len(group)})</span></div>', unsafe_allow_html=True)
        if not group:
            st.markdown('<div class="sm-empty">None yet.</div>', unsafe_allow_html=True)
            continue
        for row in range(0, len(group), 4):
            cols = st.columns(4, gap="small")
            for col, r in zip(cols, group[row:row + 4]):
                with col:
                    img = f'<img src="{esc(r["image"])}" alt=""/>' if r.get("image") else ""
                    state = ('<span class="sm-state">Loved</span>' if kind == "like"
                             else '<span class="sm-state no">Not for me</span>')
                    st.markdown(f'<div class="sm-choice">{img}<div><div class="br">{esc(r["brand"])}</div>'
                                f'<div class="nm">{esc(pretty(r["name"]))}</div>{state}</div></div>',
                                unsafe_allow_html=True)
                    other, other_label = (("not_for_me", "Move to ✕") if kind == "like" else ("like", "Move to ♡"))
                    c1, c2 = st.columns(2)
                    c1.button("↺ Undo", key=f"undo-{tab}-{r['parent_asin']}", use_container_width=True,
                              on_click=react, args=(tab, r["parent_asin"], "undo"))
                    c2.button(other_label, key=f"move-{tab}-{r['parent_asin']}", use_container_width=True,
                              on_click=react, args=(tab, r["parent_asin"], other))
    st.button("Clear all my choices", key=f"clear-{tab}", on_click=clear_all,
              args=(tab, [r["parent_asin"] for r in reactions]))


def more_like(asin: str, name: str):
    st.session_state.similar = (name, api("GET", f"/perfumes/{asin}/similar?k=6"))


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
    if resp.get("reactions"):
        n_like = sum(r["reaction"] == "like" for r in resp["reactions"])
        st.markdown(f'<div class="sm-adapt">Your selection adapts to you: {n_like} loved, '
                    f'{len(resp["reactions"]) - n_like} turned down. Changed your mind? Edit your choices below '
                    f'the selection{" or in “Your collection”" if tab == "quiz" else ""}.</div>', unsafe_allow_html=True)
    current = {r["parent_asin"]: r["reaction"] for r in resp.get("reactions") or []}
    items = resp["items"]
    for row in range(0, len(items), 3):
        cols = st.columns(3, gap="medium")
        for col, p in zip(cols, items[row:row + 3]):
            with col:
                st.markdown(card_html(p), unsafe_allow_html=True)
                with st.expander("Why this perfume?"):
                    st.markdown(why_html(p), unsafe_allow_html=True)
                asin = p["parent_asin"]
                reaction_buttons(tab, asin, f"{tab}-{asin}", current.get(asin))
                st.button("More like this", key=f"sim-{tab}-{asin}", use_container_width=True,
                          on_click=more_like, args=(asin, p["name"]))
                st.write("")
    if st.session_state.similar:
        name, sim = st.session_state.similar
        if sim:
            st.markdown(f'<div class="sm-kicker" style="margin-top:30px">More like {esc(pretty(name))}</div>',
                        unsafe_allow_html=True)
            cols = st.columns(3, gap="medium")
            for n, p in enumerate(sim["items"][:6]):
                with cols[n % 3]:
                    st.markdown(card_html(p), unsafe_allow_html=True)
                    reaction_buttons(tab, p["parent_asin"], f"sim-{tab}-{p['parent_asin']}",
                                     current.get(p["parent_asin"]))
    if resp.get("reactions") is not None:
        st.markdown('<div class="sm-sep"></div>', unsafe_allow_html=True)
        choices_panel(resp["reactions"], tab,
                      "Your choices" if tab == "quiz" else "Your choices on this profile",
                      "Nothing yet: ♡ or ✕ a perfume above.")


# ------------------------------------------------------------------ page
health = api("GET", "/health")
if "restored" not in st.session_state:
    # coming back with the same ?visitor= link: refill the quiz and rebuild the selection
    st.session_state.restored = True
    past = api("GET", f"/session/{st.session_state.session_user}") or {}
    for qid, option in (past.get("answers") or {}).items():
        st.session_state[f"q-{qid}"] = option
    st.session_state["extra"] = past.get("extra_notes") or []
    if past.get("seed_notes") or past.get("reactions"):
        refresh("quiz")
st.markdown('<div class="sm-top"><span>Maison ScentMatch</span><span>Haute parfumerie · guided by taste</span></div>'
            '<div class="sm-hero"><div class="sm-mark">SCENTMATCH</div>'
            '<div class="sm-tag">The art of finding your signature scent</div><div class="sm-rule"></div></div>',
            unsafe_allow_html=True)

mode = st.segmented_control("mode", ["Discover your scent", "Your collection"], default="Discover your scent",
                            label_visibility="collapsed", key="mode")
st.write("")

if mode != "Your collection":
    st.markdown('<div class="sm-kicker">The olfactory portrait</div><div class="sm-h2">Six questions, one signature</div>'
                '<div class="sm-lead">Answer as many as you like. Each answer becomes notes, and the notes become a selection. '
                'Then tell us what you love or not: the selection adapts at every click.</div>',
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
        extra = st.multiselect("notes", load_notes(), key="extra", placeholder="vanilla, rose, oud…", label_visibility="collapsed")
        st.write("")
        go = st.form_submit_button("Reveal my selection", type="primary", use_container_width=True)
    if go:
        answers = {k: v for k, v in answers.items() if v}
        if not answers and not extra:
            st.session_state.quiz_resp = None
            st.warning("Choose at least one answer or one note.")
        else:
            log("quiz", payload={"answers": answers, "extra_notes": extra})
            st.session_state.similar = None
            # the session endpoint = quiz + every like / rejection already given in this visit
            st.session_state.quiz_resp = api("POST", "/recommendations/session", json={
                "user_id": st.session_state.session_user, "answers": answers, "extra_notes": extra, "k": 9})
    st.write("")
    show(st.session_state.quiz_resp, "quiz")

else:
    me = api("GET", f"/session/{st.session_state.session_user}") or {}
    st.markdown('<div class="sm-kicker">This visit</div><div class="sm-h2">Your collection</div>'
                '<div class="sm-lead">Everything you told us in “Discover your scent”. Made a mistake? Undo a choice '
                'or move it to the other side: your selection is recomputed at once.</div>', unsafe_allow_html=True)
    if me.get("seed_notes"):
        chips = "".join(f'<span class="sm-chip">{esc(n)}</span>' for n in me["seed_notes"])
        answers = " · ".join(me.get("answers", {}).values())
        st.markdown(f'<div class="sm-q" style="font-size:19px">Your portrait{": " + esc(answers) if answers else ""}</div>'
                    f'<div class="sm-chips">{chips}</div>', unsafe_allow_html=True)
    choices_panel(me.get("reactions") or [], "quiz", "Your choices",
                  "Nothing yet: take the portrait in “Discover your scent”, then ♡ or ✕ the perfumes you are shown.")
    st.markdown('<div class="sm-sep"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sm-kicker">Returning client</div><div class="sm-h2">Your collection, continued</div>'
                '<div class="sm-lead">Enter your Amazon reviewer id. The NLGCL model reads the perfumes you loved '
                'and proposes what comes next.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    uid = c1.text_input("Amazon user id", placeholder="e.g. AEJI7MSPXQCTPRTWBSQYK4NU7AOQ", label_visibility="collapsed")
    ask = c2.button("Recommend", type="primary", use_container_width=True)
    demo = st.pills("Try a demo profile", list(DEMO_USERS), key="demo", label_visibility="visible")
    target = uid.strip() if ask and uid.strip() else (DEMO_USERS[demo] if demo and not ask else None)
    if target and target != (st.session_state.user_resp or {}).get("user_id"):
        st.session_state.user_resp = api("GET", f"/recommendations/{target}?k=9", quiet_404=True)
        st.session_state.similar = None
        if st.session_state.user_resp:
            refresh("user")             # picks up the reactions already given on this profile, if any
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
