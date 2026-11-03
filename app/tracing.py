#!/usr/bin/env python3
"""Opik tracing for Reel Chat (user directive, 2026-11-02: "Add tracing
from opik" / "Always add it").

Every chatbot answer becomes an Opik trace (project from OPIK_PROJECT_NAME,
default "reel-places") with spans for the LLM parse, the graph pipeline,
the LLM phrasing, and the web fallback, so questions, intents, answers,
latencies and errors are inspectable in the Opik dashboard.

Activation: tracing is ON when OPIK_API_KEY is present in the environment
(Streamlit Cloud: add it under the app's Settings → Secrets; locally:
export it or store it via the secure connector flow). Without a key this
module is a strict no-op — answers are never affected. Tracing failures
must never break an answer, so everything here fails soft.
"""
import os

# Workspace pinned in code (user directive, 2026-11-03): traces go to the
# "harikrishnad1997" workspace unless OPIK_WORKSPACE is explicitly set in
# the environment/Secrets. The project is created automatically by Opik
# when the first trace lands — nothing to provision by hand.
os.environ.setdefault("OPIK_WORKSPACE", "harikrishnad1997")
_PROJECT = os.environ.get("OPIK_PROJECT_NAME", "reel-places")
_import_ok = None


def _opik():
    global _import_ok
    if _import_ok is None:
        try:
            import opik  # noqa: F401

            _import_ok = True
        except Exception:
            _import_ok = False
    return _import_ok


def enabled():
    if os.environ.get("REEL_CHAT_OPIK", "").lower() in ("off", "0", "no"):
        return False
    return bool(os.environ.get("OPIK_API_KEY")) and _opik()


def track(name, span_type="general"):
    """Decorator factory: opik.track when enabled, identity otherwise.
    Decided at decoration (import) time; the app's env — including
    Streamlit Secrets injected at startup — is already set by then."""
    if not enabled():
        return lambda fn: fn
    try:
        import opik

        return opik.track(name=name, type=span_type, project_name=_PROJECT)
    except Exception:
        return lambda fn: fn


def status():
    """(enabled, workspace label, project) for the sidebar — the workspace
    label shows exactly where traces will land, so a wrong-workspace key is
    visible instead of silent."""
    ws = os.environ.get("OPIK_WORKSPACE") or "(key's default — not set)"
    return enabled(), ws, _PROJECT


def flush():
    """Push pending traces (Opik batches in a background thread; flushing
    at the end of an answer keeps the dashboard near-real-time)."""
    if not enabled():
        return
    try:
        import opik

        opik.flush_tracker()
    except Exception:
        pass
