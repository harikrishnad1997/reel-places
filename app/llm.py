#!/usr/bin/env python3
"""Optional LLM layer for the Reel Places chatbot.

Design rules (non-negotiable):
- The graph is the ONLY source of facts. This module never answers a
  question itself. It does two things, and only when the Muse OpenAI
  connector helper is present:
    1. parse_question(): turn a free-form question into a validated,
       structured intent drawn strictly from the graph's own vocabulary.
    2. phrase_answer(): write a SHORT summary of a finished, graph-produced
       answer. The summary is rejected if it contains any URL (links live
       only in the deterministic References section chatbot.py appends) or
       if it runs long — the user asked for summary-then-references
       (2026-11-11: "The answers are too long right now").
- Any failure here -> these functions return None and set LAST_ERROR;
  chatbot.py runs in LLM-only mode (user directive, 2026-11-02), so it
  then explains the failure instead of falling back to keyword rules.

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

import tracing

INTENTS = {"theme", "creator", "browse", "reels", "count", "itinerary", "between", "unknown"}
_URL_RE = re.compile(r"https?://[^\s\"')\]]+")
HELPER = Path(os.environ.get(
    "REEL_CHAT_OPENAI_HELPER",
    Path.home() / "workspace/custom-connectors/custom.openai/bin/openai_chat.py"))

# Short, key-free description of the most recent _chat failure (e.g.
# "OpenAI HTTP 401"), so the UI can tell the user WHY the AI call failed
# instead of a bare "couldn't understand". Never holds any credential.
LAST_ERROR = None


def available():
    if os.environ.get("REEL_CHAT_LLM", "").lower() in ("off", "0", "no"):
        return False
    return HELPER.is_file() or bool(os.environ.get("OPENAI_API_KEY"))


@tracing.track("openai-call", span_type="llm")
def _chat(system, user, temperature_note=None):
    """One chat completion; None on any failure.

    Backend order: (1) Muse's secure connector helper on this machine —
    the key is injected by surrogate exchange and never read here.
    (2) A direct OpenAI API call using OPENAI_API_KEY from the environment
    — this is the hosted path (e.g. Streamlit Cloud Secrets inject it as an
    env var). The key is read at call time only, never stored or logged.
    """
    global LAST_ERROR
    if not available():
        LAST_ERROR = "no AI connection configured"
        return None
    model = os.environ.get("REEL_CHAT_MODEL", "gpt-4o-mini")
    if HELPER.is_file():
        try:
            proc = subprocess.run(
                [sys.executable, str(HELPER), "--system", system,
                 "--user", user, "--model", model],
                capture_output=True, text=True, timeout=60)
        except Exception as exc:
            proc = None
            LAST_ERROR = f"AI helper error ({type(exc).__name__})"
        if proc is not None and proc.returncode == 0:
            out = (proc.stdout or "").strip()
            if out:
                LAST_ERROR = None
                return out
        if proc is not None:
            LAST_ERROR = "AI helper returned an error"
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        try:
            import urllib.error
            import urllib.request

            class _NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *args):
                    raise RuntimeError("redirect refused")

            payload = {"model": model, "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}]}
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"},
                method="POST")
            with urllib.request.build_opener(_NoRedirect()).open(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = (data["choices"][0]["message"]["content"] or "").strip()
            if out:
                LAST_ERROR = None
                return out
            LAST_ERROR = "OpenAI returned an empty reply"
            return None
        except urllib.error.HTTPError as exc:
            LAST_ERROR = f"OpenAI HTTP {exc.code}"
            return None
        except Exception as exc:
            LAST_ERROR = f"OpenAI call failed ({type(exc).__name__})"
            return None
    return None


@tracing.track("llm-parse", span_type="general")
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
        "(one of theme, creator, browse, reels, count, itinerary, between, unknown), "
        "theme, geo (a state, region or country name), site, creator, days "
        "(an integer or null), site_a, site_b (the two sites for a between question).\n"
        "Intent guide: a question naming a park/site and asking for its "
        "highlights, overview, spots or places is intent browse with that "
        "site. A question naming only a state/region/country is browse with "
        "that geo. Use theme only when the question asks for an activity or "
        "subject (sunrise, waterfalls, hikes) — and still fill site or geo "
        "when one is named. A question asking for places, stops or things "
        "to see BETWEEN two named parks/sites, or on the way from one to "
        "the other, is intent between with site_a and site_b set to the "
        "two sites (in the order asked) — never answer it as a browse of "
        "either endpoint alone.\n"
        "Examples:\n"
        '- "Mammoth cave highlights" -> {"intent": "browse", "theme": null, "geo": null, "site": "Mammoth Cave National Park", "creator": null, "days": null}\n'
        '- "Places in California" -> {"intent": "browse", "theme": null, "geo": "California", "site": null, "creator": null, "days": null}\n'
        '- "best sunrise spots in Utah" -> {"intent": "theme", "theme": "Sunrise", "geo": "Utah", "site": null, "creator": null, "days": null}\n'
        '- "make a 3-day itinerary for Zion" -> {"intent": "itinerary", "theme": null, "geo": null, "site": "Zion National Park", "creator": null, "days": 3}\n'
        '- "places between Joshua Tree and Death Valley" -> {"intent": "between", "theme": null, "geo": null, "site": null, "creator": null, "days": null, "site_a": "Joshua Tree National Park", "site_b": "Death Valley National Park"}'
    )
    user = f"Vocab: {json.dumps(vocab)}\n\nQuestion: {question}"
    raw = _chat(system, user)
    if not raw:
        raw = _chat(system, user)  # one retry for a transient failure
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)  # salvage a JSON object in prose
        try:
            parsed = json.loads(m.group(0)) if m else None
        except Exception:
            parsed = None
    if not isinstance(parsed, dict) or parsed.get("intent") not in INTENTS:
        return None

    def match(value, options, substring=False):
        """Exact vocab match, with a case/whitespace-insensitive rescue —
        the model sometimes returns the right value in the wrong case.
        With substring=True (sites and geos), a unique vocab entry that
        contains the value, or is contained by it, also matches — e.g.
        "Mammoth Cave" for "Mammoth Cave National Park"."""
        if value in options:
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            for opt in options:
                if opt.lower() == low:
                    return opt
            if substring and low:
                hits = [o for o in options if low in o.lower() or o.lower() in low]
                if len(hits) == 1:
                    return hits[0]
        return None

    geo_vocab = vocab.get("states", []) + vocab.get("regions", []) + vocab.get("countries", [])
    out = {"intent": parsed["intent"], "theme": None, "geo": None,
           "site": None, "creator": None, "days": None,
           "site_a": None, "site_b": None}
    out["theme"] = match(parsed.get("theme"), vocab.get("themes", []))
    out["geo"] = match(parsed.get("geo"), geo_vocab, substring=True)
    out["site"] = match(parsed.get("site"), vocab.get("sites", []), substring=True)
    out["creator"] = match(parsed.get("creator"), vocab.get("creators", []))
    out["site_a"] = match(parsed.get("site_a"), vocab.get("sites", []), substring=True)
    out["site_b"] = match(parsed.get("site_b"), vocab.get("sites", []), substring=True)
    days = parsed.get("days")
    if isinstance(days, int) and 1 <= days <= 14:
        out["days"] = days
    if out["intent"] == "unknown":
        return None
    return out


@tracing.track("web-search", span_type="general")
def web_answer(question):
    """Web-search fallback (user directive, 2026-11-02): when the graph
    can't answer, search the web and answer from it instead of dead-ending.

    Uses the OpenAI Responses API with the web_search tool, via the direct
    key path only (Streamlit Cloud / any OPENAI_API_KEY deployment). The
    connector-helper path has no web tool, so locally this returns None.
    Answers are always labelled as web-sourced by the caller. Returns the
    answer text with source URLs, or None (LAST_ERROR set) on failure."""
    global LAST_ERROR
    if not available():
        LAST_ERROR = "no AI connection configured"
        return None
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        LAST_ERROR = "web fallback needs the direct API connection"
        return None
    try:
        import urllib.error
        import urllib.request

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args):
                raise RuntimeError("redirect refused")

        payload = {
            "model": os.environ.get("REEL_CHAT_MODEL", "gpt-4o-mini"),
            "tools": [{"type": "web_search"}],
            "input": [
                {"role": "system", "content": (
                    "Answer the user's question using web search, in a "
                    "natural, conversational tone — like a knowledgeable "
                    "friend answering directly, in full flowing sentences, "
                    "never telegram fragments. Structure: (1) a short "
                    "answer first — a couple of sentences, or a short intro "
                    "plus at most 6 highlights, each a full sentence naming "
                    "the place and why it matters; (2) then a 'References:' "
                    "section listing the pages you used as Markdown links. "
                    "Keep it tight, no background essays.")},
                {"role": "user", "content": question}],
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"},
            method="POST")
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        texts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and part.get("text"):
                        texts.append(part["text"])
        out = "\n".join(texts).strip()
        if out:
            LAST_ERROR = None
            return out
        LAST_ERROR = "web search returned an empty answer"
        return None
    except urllib.error.HTTPError as exc:
        LAST_ERROR = f"OpenAI HTTP {exc.code}"
        return None
    except Exception as exc:
        LAST_ERROR = f"web search failed ({type(exc).__name__})"
        return None


@tracing.track("llm-phrase", span_type="general")
def phrase_answer(question, grounded_answer):
    """Write a SHORT summary of a graph-produced answer, or None.

    User format (2026-11-11): the bot gives a summary first, then
    references — full answers were too long. So this returns ONLY the
    summary; chatbot.py appends a deterministic References section built
    from the grounded answer itself, so no source can be dropped or
    invented. The summary is rejected if it contains any URL (links live
    in References only) or if it exceeds the length cap.
    """
    if not available() or not grounded_answer:
        return None
    system = (
        "Rewrite the answer below as a natural, conversational reply — "
        "like a knowledgeable friend answering the question directly. Rules:\n"
        "- Write in full, flowing sentences. Never use telegram fragments "
        "or 'Name: short phrase' list-speak.\n"
        "- Answer the question first in a sentence or two, then work the "
        "places in naturally. For list questions, a short intro line plus "
        "at most 8 bullets, each bullet a full sentence naming the place "
        "and why it matters.\n"
        "- If the answer is an itinerary, keep the day-by-day structure, "
        "but write each day as one or two natural sentences instead of "
        "compressed stop lines.\n"
        "- Keep it tight (about 100-180 words) and cut anything that "
        "repeats the same point.\n"
        "- Use ONLY facts present in the answer; never add places, "
        "names or facts.\n"
        "- Include NO links or URLs at all — references are appended "
        "separately.\n"
        "Output only the reply."
    )
    user = f"Question: {question}\n\nAnswer to summarize:\n{grounded_answer}"
    out = _chat(system, user)
    if not out:
        return None
    out = out.strip()
    if _URL_RE.search(out):
        return None  # links belong in References only — keep the original
    if not out or len(out) < 10 or len(out) > 2500:
        return None  # empty, or still too long — keep the original
    return out
