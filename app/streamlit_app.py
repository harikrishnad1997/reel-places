#!/usr/bin/env python3
"""Reel Chat — Streamlit front end for the Reel Places graph chatbot.

Run from this folder:
    .venv/bin/streamlit run streamlit_app.py

It reuses chatbot.py's engine (Kuzu graph DB, graph.json fallback) and the
optional OpenAI LLM layer in llm.py. The OpenAI key is never read here: the
LLM layer calls Muse's custom.openai connector helper, which injects the
credential by surrogate exchange at request time. Nothing secret is in this
file, the browser, or the repo.
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
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chatbot  # noqa: E402
import llm  # noqa: E402

st.set_page_config(page_title="Reel Chat — Reel Places", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading the reel graph…")
def get_graph():
    data, backend_note = chatbot.load_graph()
    return chatbot.Graph(data), backend_note, data


g, backend_note, data = get_graph()

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
    st.caption("LLM layer: " + ("on (OpenAI via secure connector)" if llm.available() else "off — rule-based answers only"))
    st.caption("Answers come only from the graph. Every place cites its source reel.")

st.title("Reel Chat")
st.write("Ask about the places saved from Instagram reels, or build an itinerary from their recommendations.")

if not llm.available():
    st.error(
        "This chatbot runs in **LLM-only mode** (no rule-based fallback), and no AI "
        "connection is configured for this deployment yet. To switch it on, add "
        "`OPENAI_API_KEY` under the app's Settings → Secrets in Streamlit Cloud, "
        "then reboot the app."
    )
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
