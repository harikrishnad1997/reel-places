#!/usr/bin/env python3
"""Reel Chat — Streamlit front end for the Reel Places graph chatbot.

Run from this folder:
    .venv/bin/streamlit run streamlit_app.py

It reuses chatbot.py's engine (Kuzu graph DB, graph.json fallback) and the
OpenAI LLM layer in llm.py (LLM-only — no rule-based fallback). Where the
key comes from depends on the deployment: locally, Muse's custom.openai
connector helper injects it by surrogate exchange; on Streamlit Cloud it
can be set as an app Secret, or the user can paste it into the page's key
field, where it is held only in that browser session's memory and sent
only to OpenAI. The key is never stored in this file or the repo.
"""
import os
import sys
from pathlib import Path

import streamlit as st

# Hosted deployments (Streamlit Cloud) provide the OpenAI key as an app
# secret; expose it to the LLM layer as an env var before importing it.
# Locally there is no secrets file and the secure connector helper is used
# instead — no key is ever read from this file or the repo.
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
    # Opik tracing (always added, per the user): activates when its key is
    # present in Secrets; without one, tracing is a no-op.
    for _k in ("OPIK_API_KEY", "OPIK_WORKSPACE", "OPIK_PROJECT_NAME"):
        if _k in st.secrets:
            os.environ.setdefault(_k, st.secrets[_k])
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chatbot  # noqa: E402
import llm  # noqa: E402
import tracing  # noqa: E402

st.set_page_config(page_title="Reel Chat — Reel Places", page_icon="🎬", layout="centered")


def _graph_fingerprint():
    """mtime+size of the bundled graph.json — passed into the cached loader
    so a data update busts the cache on the next run. (Streamlit Cloud kept
    serving a stale cached graph, 144 places vs 161 in the repo, until the
    app was rebooted; this makes reboots unnecessary for data updates.)"""
    try:
        s = (HERE / "graph.json").stat()
        return f"{s.st_mtime_ns}-{s.st_size}"
    except Exception:
        return "none"


@st.cache_resource(show_spinner="Loading the reel graph…")
def get_graph(fingerprint):
    data, backend_note = chatbot.load_graph()
    return chatbot.Graph(data), backend_note, data


g, backend_note, data = get_graph(_graph_fingerprint())

places = [n for n in data["nodes"] if n["type"] == "Place"]
reels = [n for n in data["nodes"] if n["type"] == "Reel"]
creators = [n for n in data["nodes"] if n["type"] == "Creator"]
regions = {}
for p in places:
    r = p["properties"].get("region", "?")
    regions[r] = regions.get(r, 0) + 1

with st.sidebar:
    st.title("🎬 Reel Places")
    st.metric("Places", len(places))
    st.metric("Source reels", len(reels))
    st.metric("Creators", len(creators))
    st.subheader("Places by region")
    for name, count in sorted(regions.items(), key=lambda kv: -kv[1]):
        st.write(f"**{name}** · {count}")
    st.divider()
    st.caption(f"Graph backend: {backend_note}")
    st.caption("LLM layer: " + ("on — LLM-only mode" if llm.available() else "off — paste an OpenAI key on the page to chat (LLM-only, no fallback)"))
    _t_on, _t_ws, _t_proj = tracing.status()
    st.caption("Tracing (Opik): " + (f"on — workspace {_t_ws} · project {_t_proj}" if _t_on else "off — add OPIK_API_KEY to Secrets to enable"))
    st.caption("Answers come from the graph first (every place cites its source reel); if the graph doesn't cover a question, the bot searches the web and labels the answer as web-sourced.")

st.title("Reel Chat")
st.write("Ask about the places saved from Instagram reels, or build an itinerary from their recommendations.")

if not llm.available():
    st.info(
        "Reel Chat runs in **LLM-only mode** — it needs an OpenAI API key to answer. "
        "Paste yours below to start chatting. The key stays in this browser session "
        "only: it is never saved, never written to the repo, and is sent only to OpenAI."
    )
    entered_key = st.text_input(
        "OpenAI API key",
        type="password",
        key="openai_key_input",
        placeholder="sk-…",
        help="Held in this session's memory only. Alternatively, set OPENAI_API_KEY once under the app's Settings → Secrets in Streamlit Cloud.",
    )
    if entered_key and entered_key.strip():
        os.environ["OPENAI_API_KEY"] = entered_key.strip()
        st.rerun()
    st.stop()

EXAMPLES = [
    "Best sunrise spots in Utah",
    "Best waterfall spots",
    "What did @amorywild recommend in Zion?",
    "Places in California",
    "Make a 2-day itinerary for Utah",
    "How many places do we have?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []

cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, key=f"ex{i}"):
        st.session_state.pending = example

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about a place, a theme, a creator… or 'make a 3-day Southwest itinerary'")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the graph…"):
            reply = chatbot.answer(g, question, backend_note)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
