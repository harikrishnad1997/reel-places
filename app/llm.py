#!/usr/bin/env python3
"""Optional LLM layer for the Reel Places chatbot.

Design rules (non-negotiable):
- The graph is the ONLY source of facts. This module never answers a
  question itself. It does two things, and only when the Muse OpenAI
  connector helper is present:
    1. parse_question(): turn a free-form question into a validated,
       structured intent drawn strictly from the graph's own vocabulary.
    2. phrase_answer(): rephrase a finished, graph-produced answer for
       readability. The result is rejected unless it preserves every URL
       from the grounded answer and introduces no new URLs.
- Helper missing, helper erroring, or any validation failure -> these
  functions return None and chatbot.py runs its rule-based pipeline
  exactly as if this module did not exist.

How the LLM is reached (no raw key anywhere):
- All calls go through Muse's secure connector helper:
      ~/workspace/custom-connectors/custom.openai/bin/openai_chat.py
          --system "..." --user "..." [--model gpt-4o-mini]
  It prints the assistant's reply to stdout. The OpenAI API key is
  injected by a surrogate exchange at request time inside the connector;
  the key never appears in an environment variable, a file, this code, or
  any output. Override the helper path with REEL_CHAT_OPENAI_HELPER and
  the model with REEL_CHAT_MODEL (default gpt-4o-mini). Disable the layer
  entirely with REEL_CHAT_LLM=off.
- SECURITY: because chat.html is published on a public GitHub Pages site,
  no credential of any kind may ever be embedded in it. The web prototype
  is therefore rule-based only; the LLM layer exists solely in this
  local CLI path.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

INTENTS = {"theme", "creator", "browse", "reels", "count", "itinerary", "unknown"}
_URL_RE = re.compile(r"https?://[^\s\"')\]]+")
HELPER = Path(os.environ.get(
    "REEL_CHAT_OPENAI_HELPER",
    Path.home() / "workspace/custom-connectors/custom.openai/bin/openai_chat.py"))


def available():
    if os.environ.get("REEL_CHAT_LLM", "").lower() in ("off", "0", "no"):
        return False
    return HELPER.is_file()


def _chat(system, user, temperature_note=None):
    """One chat completion via the connector helper; None on any failure."""
    if not available():
        return None
    model = os.environ.get("REEL_CHAT_MODEL", "gpt-4o-mini")
    try:
        proc = subprocess.run(
            [sys.executable, str(HELPER), "--system", system,
             "--user", user, "--model", model],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def parse_question(question, vocab):
    """Free-form question -> validated intent dict, or None.

    vocab: {"themes": [...], "states": [...], "regions": [...],
            "countries": [...], "sites": [...], "creators": [...]}
    Every returned value is checked against vocab; anything the model
    invents is dropped, and an unrecognised intent yields None.
    """
    system = (
        "You convert a question about a travel-reel knowledge graph into a "
        "JSON intent. Use ONLY values copied exactly from the vocab lists "
        "provided; if a value is not in the lists, use null. Never invent "
        "places, people or themes. Output JSON only, with keys: intent "
        "(one of theme, creator, browse, reels, count, itinerary, unknown), "
        "theme, geo (a state, region or country name), site, creator, days "
        "(an integer or null)."
    )
    user = f"Vocab: {json.dumps(vocab)}\n\nQuestion: {question}"
    raw = _chat(system, user)
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if not isinstance(parsed, dict) or parsed.get("intent") not in INTENTS:
        return None
    geo_vocab = vocab.get("states", []) + vocab.get("regions", []) + vocab.get("countries", [])
    out = {"intent": parsed["intent"], "theme": None, "geo": None,
           "site": None, "creator": None, "days": None}
    if parsed.get("theme") in vocab.get("themes", []):
        out["theme"] = parsed["theme"]
    if parsed.get("geo") in geo_vocab:
        out["geo"] = parsed["geo"]
    if parsed.get("site") in vocab.get("sites", []):
        out["site"] = parsed["site"]
    if parsed.get("creator") in vocab.get("creators", []):
        out["creator"] = parsed["creator"]
    days = parsed.get("days")
    if isinstance(days, int) and 1 <= days <= 14:
        out["days"] = days
    if out["intent"] == "unknown":
        return None
    return out


def phrase_answer(question, grounded_answer):
    """Rephrase a graph-produced answer, or None to keep the original.

    The rewrite is accepted only if it keeps exactly the grounded set of
    URLs: none dropped, none added. Facts therefore cannot drift.
    """
    if not available() or not grounded_answer:
        return None
    system = (
        "Rewrite the answer below so it reads naturally. Hard rules: use "
        "ONLY facts present in the answer; never add places, recommendations, "
        "names or facts; keep every URL exactly as written, character for "
        "character; keep the source attribution attached to each place. "
        "Output only the rewritten answer."
    )
    user = f"Question: {question}\n\nAnswer to rewrite:\n{grounded_answer}"
    out = _chat(system, user)
    if not out:
        return None
    out = out.strip()
    grounded_urls = set(_URL_RE.findall(grounded_answer))
    out_urls = set(_URL_RE.findall(out))
    if grounded_urls != out_urls:
        return None  # a link was dropped or invented — keep the original
    if not out or len(out) < 10:
        return None
    return out
