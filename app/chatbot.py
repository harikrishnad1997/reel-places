#!/usr/bin/env python3
"""Reel Places chatbot - answers ONLY from the Reel Places graph.

Backend order:
  1. The Kuzu graph DB in graph_db/ (queried with Cypher via the local venv).
  2. graph.json (the exact export the DB was built from) if the DB or the
     kuzu package is unavailable.

Usage:
  ./.venv/bin/python chatbot.py "best sunrise spots"   (one-shot)
  ./.venv/bin/python chatbot.py                        (interactive)
Plain python3 also works - it falls back to graph.json.

No places are ever invented: every place named in an answer comes from a
Place node, and every place is printed with its source reel(s) and URL(s).
"""
import json
import math
import pathlib
import re
import sys

import tracing

HERE = pathlib.Path(__file__).resolve().parent


# ------------------------------------------------------------------ loading
def load_from_kuzu():
    import kuzu  # ImportError -> caller falls back to graph.json

    db_path = HERE / "graph_db"
    if not db_path.exists():
        raise FileNotFoundError("graph_db/ not found - run build_graph.py first")
    try:
        db = kuzu.Database(str(db_path), read_only=True)
    except TypeError:
        db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)
    nodes, edges = [], []
    for t in ["Place", "Site", "State", "Region", "Country", "Reel", "Creator", "Theme"]:
        res = conn.execute(f"MATCH (n:{t}) RETURN n.id, n.props")
        while res.has_next():
            nid, props = res.get_next()
            nodes.append({"id": nid, "type": t, "properties": json.loads(props)})
    for t in ["LOCATED_IN", "IN_REGION", "IN_COUNTRY", "PART_OF",
              "RECOMMENDED_BY", "CREATED_BY", "BEST_FOR", "HAS_THEME"]:
        res = conn.execute(f"MATCH (a)-[e:{t}]->(b) RETURN a.id, b.id, e.props")
        while res.has_next():
            frm, to, props = res.get_next()
            edges.append({"from": frm, "to": to, "type": t, "properties": json.loads(props)})
    conn.close()
    return {"nodes": nodes, "edges": edges}, "Kuzu graph DB (graph_db/)"


def load_graph():
    try:
        return load_from_kuzu()
    except Exception as exc:  # DB missing / kuzu missing / schema drift
        data = json.loads((HERE / "graph.json").read_text(encoding="utf-8"))
        return data, f"graph.json (DB fallback: {type(exc).__name__})"


class Graph:
    def __init__(self, data):
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.edges = data["edges"]
        self.by_type = {}
        for n in data["nodes"]:
            self.by_type.setdefault(n["type"], []).append(n)
        self.out = {}
        for e in self.edges:
            self.out.setdefault((e["from"], e["type"]), []).append(e)

    def targets(self, frm, etype):
        return [self.nodes[e["to"]] for e in self.out.get((frm, etype), []) if e["to"] in self.nodes]

    def sources(self, place_id):
        es = self.out.get((place_id, "RECOMMENDED_BY"), [])
        return sorted(es, key=lambda e: (e["properties"].get("rank") is None,
                                         e["properties"].get("rank") or 0))

    def reel_node(self, edge):
        return self.nodes.get(edge["to"], {})


# ------------------------------------------------------------------ helpers
GENERIC_SITE_WORDS = {"national", "park", "preserve", "and", "the", "of", "recreation",
                      "area", "monument", "state", "forest", "historical", "site"}

THEME_SYNONYMS = [
    (r"sunrises?\b", "Sunrise"), (r"sunsets?\b", "Sunset"),
    (r"waterfalls?\b", "Waterfall"), (r"\bfalls\b", "Waterfall"),
    (r"short hikes?\b", "Short hike"), (r"hikes?\b|hiking\b", "Hike"),
    (r"magic|filmed", "Filmed location"), (r"solitude", "Solitude"),
    (r"bucket list", "Bucket list"), (r"winter", "Winter visit"),
    (r"wildlife", "Wildlife"), (r"must[- ]see", "Must-see spot"),
    (r"top national parks?\b", "Top national park"), (r"views?\b", "View"),
]


def theme_id(name):
    return "theme:" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def find_themes(g, q):
    found = []
    for pat, name in THEME_SYNONYMS:
        if re.search(pat, q) and theme_id(name) in g.nodes and name not in found:
            found.append(name)
    return found


def site_distinctive_tokens(site_name):
    return [w for w in re.findall(r"[a-z']+", site_name.lower())
            if w not in GENERIC_SITE_WORDS and len(w) > 2]


def find_sites(g, q):
    """Sites named by the question, best matches only.

    Scoring: a full site name in the question outranks everything; below
    that, sites are scored by how many of their distinctive tokens appear,
    and only the top score is kept. (The old any-token match dragged
    Cave-in-Rock into a "Mammoth Cave" answer and Kobuk Valley into a
    "Death Valley" one — both share a single generic token.) A token-only
    match now needs at least TWO distinctive tokens: a single one is too
    weak, and theme words collide with site names — "sunrise" alone used
    to match the site "Mount Rainier National Park (Sunrise)" and veto
    every Utah sunrise hit in answer_theme (found 2026-11-11). Canonical
    questions from the LLM layer always carry full site names, so the
    two-token floor costs nothing there."""
    words = set(re.findall(r"[a-z']+", q))
    scored = []
    for s in g.by_type.get("Site", []):
        name = s["properties"]["name"].lower()
        if name in q:
            scored.append((1000 + len(name), s))
            continue
        toks = site_distinctive_tokens(name)
        score = sum(1 for tok in toks if tok in words)
        if score >= 2:
            scored.append((score, s))
    if not scored:
        return []
    top = max(score for score, _ in scored)
    return [s for score, s in scored if score == top]


def find_geo(g, q, raw_q):
    """Return (kind, node) for a state / region / country in the question.

    Longest names first, on word boundaries — otherwise "West" would match
    inside "Southwest" and "Virginia" inside "West Virginia".
    """
    def longest_first(nodes):
        return sorted(nodes, key=lambda n: -len(n["properties"]["name"]))

    def mentioned(node):
        name = re.escape(node["properties"]["name"].lower())
        return re.search(rf"(?<![a-z]){name}(?![a-z])", q) is not None

    for s in longest_first(g.by_type.get("State", [])):
        if mentioned(s):
            return "state", s
    m = re.search(r"\b([A-Z]{2})\b", raw_q)
    if m:
        for s in g.by_type.get("State", []):
            if (s["properties"].get("state_code") or "").upper() == m.group(1):
                return "state", s
    for r in longest_first(g.by_type.get("Region", [])):
        if mentioned(r):
            return "region", r
    for c in longest_first(g.by_type.get("Country", [])):
        if mentioned(c):
            return "country", c
    return None, None


def place_in_geo(place_props, kind, node):
    if kind == "state":
        return place_props.get("state") == node["properties"]["name"]
    if kind == "region":
        return place_props.get("region") == node["properties"]["name"]
    if kind == "country":
        return place_props.get("country") == node["properties"]["name"]
    return True


def fmt_place(g, p, edge_filter=None):
    pr = p["properties"]
    head = f"- {pr['name']} - {pr.get('site') or pr.get('city') or ''}"
    loc = ", ".join(x for x in [pr.get("city"), pr.get("state")] if x)
    if loc:
        head += f" ({loc}" + (f", {pr['region']}" if pr.get("region") else "") + ")"
    lines = [head]
    for e in g.sources(p["id"]):
        if edge_filter and not edge_filter(e):
            continue
        ep = e["properties"]
        reel = g.reel_node(e)
        rp = reel.get("properties", {})
        rank = f" - ranked #{ep['rank']}" if ep.get("rank") else ""
        bf = f"Best for {ep['best_for']}: " if ep.get("best_for") else ""
        rec = ep.get("recommendation") or ""
        lines.append(f"  {bf}{rec}{rank}" if rec else f"  {bf.strip()}{rank}")
        title = rp.get("title") or ep.get("reel_title") or "Reel"
        creator = rp.get("creator") or ep.get("creator") or ""
        lines.append(f'  Source: "{title}" by {creator} - {rp.get("url") or ""}')
    return "\n".join(lines)


def no_match(what):
    return (f"I don't have {what} in the Reel Places graph yet - nothing in the "
            f"reels added so far covers it, and I only answer from what's in the graph.")


# ------------------------------------------------------------------ intents
def answer_count(g):
    places = g.by_type.get("Place", [])
    reels = g.by_type.get("Reel", [])
    creators = g.by_type.get("Creator", [])
    lines = [f"The graph currently holds {len(places)} places from "
             f"{len(reels)} reels by {len(creators)} creators."]
    by_region = {}
    for p in places:
        r = p["properties"].get("region") or "Unknown"
        by_region[r] = by_region.get(r, 0) + 1
    for r, n in sorted(by_region.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {r}: {n}")
    lines.append(f"({len(g.nodes)} nodes and {len(g.edges)} edges in total.)")
    return "\n".join(lines)


def answer_reels(g):
    reels = g.by_type.get("Reel", [])
    if not reels:
        return no_match("any reels")
    lines = [f"{len(reels)} reels in the graph:"]
    for r in sorted(reels, key=lambda x: x["properties"].get("title") or ""):
        rp = r["properties"]
        lines.append(f'- "{rp.get("title")}" by {rp.get("creator")} - theme: '
                     f"{rp.get('theme') or 'n/a'}, {rp.get('places_count') or '?'} place(s)")
        lines.append(f"  {rp.get('url')}")
    return "\n".join(lines)


def answer_creator(g, q, raw_q):
    m = re.search(r"@[A-Za-z0-9._]+", raw_q)
    handle = m.group(0).lower() if m else None
    creator = None
    for c in g.by_type.get("Creator", []):
        cp = c["properties"]
        if handle and cp.get("handle", "").lower() == handle:
            creator = c
            break
        if not handle and cp.get("handle", "").lstrip("@").lower() in q:
            creator = c
            break
    if not creator:
        names = ", ".join(c["properties"]["handle"] for c in g.by_type.get("Creator", []))
        return f"I can't find that creator in the graph. Creators so far: {names}."
    handle = creator["properties"]["handle"]
    kind, geo = find_geo(g, q, raw_q)
    sites = find_sites(g, q)
    hits = []
    for p in g.by_type.get("Place", []):
        if not any((e["properties"].get("creator") or "") == handle for e in g.sources(p["id"])):
            continue
        if kind and not place_in_geo(p["properties"], kind, geo):
            continue
        if sites and p["properties"].get("site") not in [s["properties"]["name"] for s in sites]:
            continue
        hits.append(p)
    scope = ""
    if kind:
        scope = f" in {geo['properties']['name']}"
    elif sites:
        scope = " in " + ", ".join(s["properties"]["name"] for s in sites)
    if not hits:
        return no_match(f"any recommendations from {handle}{scope}")
    lines = [f"{handle} recommended {len(hits)} place(s) in the graph{scope}:"]
    for p in hits:
        lines.append(fmt_place(g, p, edge_filter=lambda e, h=handle: e["properties"].get("creator") == h))
    return "\n".join(lines)


def answer_theme(g, q, raw_q):
    themes = find_themes(g, q)
    if not themes:
        return None
    kind, geo = find_geo(g, q, raw_q)
    sites = find_sites(g, q)
    out = []
    for theme in themes:
        hits = []
        for p in g.by_type.get("Place", []):
            theme_names = [t["properties"]["name"] for t in g.targets(p["id"], "BEST_FOR")]
            if theme not in theme_names:
                continue
            if kind and not place_in_geo(p["properties"], kind, geo):
                continue
            if sites and p["properties"].get("site") not in [s["properties"]["name"] for s in sites]:
                continue
            hits.append(p)
        scope = ""
        if kind:
            scope = f" in {geo['properties']['name']}"
        elif sites:
            scope = " in " + ", ".join(s["properties"]["name"] for s in sites)
        if not hits:
            out.append(no_match(f"any {theme.lower()} spots{scope}"))
            continue
        def rank_key(p):
            ranks = [e["properties"]["rank"] for e in g.sources(p["id"]) if e["properties"].get("rank")]
            return (min(ranks) if ranks else 99, p["properties"]["name"])
        hits.sort(key=rank_key)
        lines = [f"{len(hits)} {theme.lower()} spot(s){scope}:"]
        lines += [fmt_place(g, p) for p in hits]
        out.append("\n".join(lines))
    return "\n\n".join(out)


def answer_between(g, site_a, site_b):
    """Places BETWEEN two sites (user, 2026-11-11: 'places between Joshua
    Tree and Death Valley' came back as Death Valley's own highlights —
    the between-constraint was ignored entirely). Each anchor resolves to
    the centroid of its places; candidates are graph places within 50
    miles of the straight segment between the centroids, excluding both
    anchors' own places and anything within 20 miles of either anchor
    (endpoint neighbourhood, not the corridor). Ordered along the route,
    capped at 12. An empty corridor returns no_match(...), which the
    caller turns into the labelled web fallback instead of substituting
    one endpoint's highlights."""
    def display(site):
        name = (site or "").strip().lower()
        for s in g.by_type.get("Site", []):
            if s["properties"]["name"].lower() == name:
                return s["properties"]["name"]
        return site
    site_a, site_b = display(site_a), display(site_b)

    def anchor(site):
        name = (site or "").strip().lower()
        hits = [p for p in g.by_type.get("Place", [])
                if (p["properties"].get("site") or "").lower() == name
                and p["properties"].get("lat") is not None]
        if not hits:
            return None
        return (sum(p["properties"]["lat"] for p in hits) / len(hits),
                sum(p["properties"]["lng"] for p in hits) / len(hits))

    a = anchor(site_a)
    b = anchor(site_b)
    if not a or not b:
        return no_match(f"places between {site_a} and {site_b}")
    mx = math.cos(math.radians((a[0] + b[0]) / 2.0))

    def seg(p):
        ax, ay = a[1] * mx, a[0]
        bx, by = b[1] * mx, b[0]
        px, py = p[1] * mx, p[0]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy)) * 69.0, t

    anchors = {site_a.strip().lower(), site_b.strip().lower()}
    cands = []
    for p in g.by_type.get("Place", []):
        pr = p["properties"]
        if (pr.get("site") or "").lower() in anchors:
            continue
        if pr.get("lat") is None or pr.get("lng") is None:
            continue
        d, t = seg((pr["lat"], pr["lng"]))
        if d > 50 or t <= 0.05 or t >= 0.95:
            continue
        if haversine((pr["lat"], pr["lng"]), a) < 32.2:  # within 20 mi of anchor A
            continue
        if haversine((pr["lat"], pr["lng"]), b) < 32.2:  # within 20 mi of anchor B
            continue
        cands.append((t, p))
    cands.sort(key=lambda x: x[0])
    if not cands:
        return no_match(f"any places between {site_a} and {site_b}")
    lines = [f"{len(cands)} place(s) between {site_a} and {site_b}, in route order:"]
    lines += [fmt_place(g, p) for _, p in cands[:12]]
    return "\n".join(lines)


def answer_browse(g, q, raw_q):
    sites = find_sites(g, q)
    kind, geo = find_geo(g, q, raw_q)
    if sites:
        names = [s["properties"]["name"] for s in sites]
        hits = [p for p in g.by_type.get("Place", []) if p["properties"].get("site") in names]
        if not hits:
            return no_match(f"any places in {', '.join(names)}")
        lines = [f"{len(hits)} place(s) in {', '.join(names)}:"]
        lines += [fmt_place(g, p) for p in sorted(hits, key=lambda p: p["properties"]["name"])]
        return "\n".join(lines)
    if kind:
        hits = [p for p in g.by_type.get("Place", []) if place_in_geo(p["properties"], kind, geo)]
        if not hits:
            return no_match(f"any places in {geo['properties']['name']}")
        lines = [f"{len(hits)} place(s) in {geo['properties']['name']}:"]
        lines += [fmt_place(g, p) for p in sorted(hits, key=lambda p: p["properties"]["name"])]
        return "\n".join(lines)
    return None


# ------------------------------------------------------------------ itinerary
def haversine(a, b):
    r = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


MAX_STOPS_PER_DAY = 4


def _site_group(site):
    """Group site variants under one park: 'Mount Rainier National Park
    (Sunrise)' -> 'Mount Rainier National Park'. Used only for itinerary
    grouping, so area variants of one park never count as separate parks."""
    if not site:
        return "Other"
    return re.sub(r"\s*\([^)]*\)\s*$", "", site).strip() or site


def answer_itinerary(g, q, raw_q):
    explicit_days = bool(re.search(
        r"(\d+)\s*[- ]?\s*days?\b|weekend", (raw_q or "").lower()))
    m = re.search(r"(\d+)\s*[- ]?\s*days?\b", q)
    days = int(m.group(1)) if m else (2 if "weekend" in q else 3 if "road trip" in q else 2)
    days = max(1, min(days, 14))

    sites = find_sites(g, q)
    kind, geo = find_geo(g, q, raw_q)
    if sites:
        names = [s["properties"]["name"] for s in sites]
        pool = [p for p in g.by_type.get("Place", []) if p["properties"].get("site") in names]
        scope_name = " and ".join(names)
    elif kind:
        pool = [p for p in g.by_type.get("Place", []) if place_in_geo(p["properties"], kind, geo)]
        scope_name = geo["properties"]["name"]
    else:
        pool = list(g.by_type.get("Place", []))
        scope_name = "all Reel Places"
    pool = [p for p in pool if p["properties"].get("lat") is not None]
    if not pool:
        return no_match(f"any places for an itinerary in {scope_name}")

    # "Cover all NPs in Washington" means the national parks, not every
    # place in the state: restrict the pool to national-park sites and
    # say so. Falls back to the full pool when the scope holds no park.
    parks_only = False
    excluded_non_park = 0
    if re.search(r"national parks?\b|\bnps?\b", (raw_q or "").lower()):
        np_pool = [p for p in pool
                   if "national park" in _site_group(
                       p["properties"].get("site") or "").lower()]
        if np_pool:
            excluded_non_park = len(pool) - len(np_pool)
            pool = np_pool
            parks_only = True
            if not scope_name.lower().endswith("national parks"):
                scope_name = scope_name + " national parks"

    # Group by park; order parks by greedy nearest-neighbour starting from
    # the park with the most places, so days follow geography.
    by_site = {}
    for p in pool:
        key = _site_group(p["properties"].get("site")
                          or p["properties"].get("city") or "Other")
        by_site.setdefault(key, []).append(p)
    centroids = {s: (sum(p["properties"]["lat"] for p in ps) / len(ps),
                     sum(p["properties"]["lng"] for p in ps) / len(ps))
                 for s, ps in by_site.items()}
    remaining = set(by_site)
    start = max(remaining, key=lambda s: (len(by_site[s]), s))
    order, cur = [start], start
    remaining.remove(start)
    while remaining:
        nxt = min(remaining, key=lambda s: haversine(centroids[cur], centroids[s]))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    stream = []
    for s in order:
        stream.extend(sorted(by_site[s], key=lambda p: p["properties"]["name"]))

    # No trip length given: the shortest plan that touches every park is
    # one day per park (clamped), not the old 2-day default — 2 days can
    # never cover 3 far-apart parks, and the old packing silently dropped
    # every park but the biggest (found 2026-11-12: "cover all NPs in
    # Washington" returned Olympic only).
    days_defaulted = False
    if not explicit_days:
        days = max(2, min(len(by_site), 7))
        days_defaulted = True

    overflow = []
    rescued = []
    if 1 < len(order) <= days:
        # One park per day, in geographic order; a park's extra stops
        # overflow. Extra days (beyond one per park) split the fullest
        # days instead of mixing parks into a day.
        plan = []
        for s in order:
            ps = sorted(by_site[s], key=lambda p: p["properties"]["name"])
            plan.append(ps[:MAX_STOPS_PER_DAY])
            overflow.extend(ps[MAX_STOPS_PER_DAY:])
        while len(plan) < days:
            fullest = max(range(len(plan)), key=lambda i: len(plan[i]))
            if len(plan[fullest]) < 2:
                break
            day = plan[fullest]
            half = len(day) // 2
            plan[fullest:fullest + 1] = [day[:half], day[half:]]
    else:
        # Pack the geographically ordered stream into days: keep a site
        # together where it fits, cap stops per day (the final day absorbs
        # any remainder).
        plan, day = [], []
        for s in order:
            ps = sorted(by_site[s], key=lambda p: p["properties"]["name"])
            if day and len(day) + len(ps) > MAX_STOPS_PER_DAY:
                plan.append(day)
                day = []
            for p in ps:
                if len(day) >= MAX_STOPS_PER_DAY:
                    plan.append(day)
                    day = []
                day.append(p)
        if day:
            plan.append(day)
        # More days than asked for at the per-day cap: keep the requested
        # days sane and list the rest as not fitted, instead of overloading
        # a day.
        if len(plan) > days:
            overflow = [p for d in plan[days:] for p in d]
            plan = plan[:days]
        # If the site-packing produced fewer days than asked and there are
        # enough places, re-chunk the ordered stream evenly so every day
        # is used — but never above the per-day cap.
        if len(plan) < days and len(stream) >= days:
            per = math.ceil(len(stream) / days)
            if per <= MAX_STOPS_PER_DAY:
                plan = [stream[i:i + per] for i in range(0, len(stream), per)]
                overflow = []
        # Coverage rescue: truncating a site-major stream keeps only the
        # first park(s). Every national park in scope gets at least one
        # stop in the plan, by swapping a rescued stop in for a stop from
        # the most-represented park of the most monotonous day.
        plan_sites = {_site_group(p["properties"].get("site") or "")
                      for d in plan for p in d}
        missing_parks = [s for s in order
                         if "national park" in s.lower() and s not in plan_sites]
        # Only rescue when every missing park can actually fit: with more
        # missing parks than stops, rescue is meaningless (e.g. a
        # scope-less all-places plan) — leave the plan as packed.
        if len(missing_parks) > days * MAX_STOPS_PER_DAY:
            missing_parks = []
        for s in missing_parks:
            first = sorted(by_site[s],
                           key=lambda p: p["properties"]["name"])[0]
            short = [d for d in plan if len(d) < MAX_STOPS_PER_DAY]
            if short:
                short[-1].append(first)
                rescued.append(s)
                continue
            def _mono(d):
                counts = {}
                for p in d:
                    k = _site_group(p["properties"].get("site") or "")
                    counts[k] = counts.get(k, 0) + 1
                return max(counts.values()) if counts else 0
            day = max(plan, key=lambda d: (_mono(d), len(d)))
            counts = {}
            for p in day:
                k = _site_group(p["properties"].get("site") or "")
                counts[k] = counts.get(k, 0) + 1
            majority = max(counts, key=lambda k: (counts[k], k))
            idx = max(i for i, p in enumerate(day)
                      if _site_group(p["properties"].get("site") or "") == majority)
            displaced = day[idx]
            day[idx] = first
            overflow.insert(0, displaced)
            rescued.append(s)

    def slot_of(p):
        themes = [t["properties"]["name"] for t in g.targets(p["id"], "BEST_FOR")]
        bf = (p["properties"].get("best_for") or "").lower()
        if "Sunrise" in themes or "sunrise" in bf:
            return 0
        if "Sunset" in themes or "sunset" in bf:
            return 2
        return 1

    slot_names = {0: "Morning (sunrise pick)", 1: "Daytime", 2: "Evening (sunset pick)"}
    lines = [f"{len(plan)}-day itinerary - {scope_name} "
             f"({len(pool)} place(s) from the reels, grouped by park and ordered by proximity):"]
    if parks_only and excluded_non_park:
        lines.append(f"National parks only: {excluded_non_park} more place(s) "
                     f"in this scope are outside the national parks and are not "
                     f"in this plan - ask for the full "
                     f"{scope_name.removesuffix(' national parks')} itinerary "
                     f"to include them.")
    if days_defaulted:
        lines.append(f"No trip length was given, so this is the shortest plan "
                     f"that touches every park here - {len(plan)} day(s). Add "
                     f"days to fit more of the {len(overflow)} other place(s) "
                     f"in scope.")
    if rescued:
        if len(rescued) == 1:
            lines.append(f"With {days} day(s) the parks share days, so "
                         f"{rescued[0]} only gets a single stop - give the "
                         f"trip more days for a full day per park.")
        else:
            lines.append(f"With {days} day(s) the parks share days, so "
                         f"{', '.join(rescued)} only get a single stop each - "
                         f"give the trip more days for a full day per park.")
    for i, day_places in enumerate(plan, 1):
        day_sites = []
        for p in day_places:
            s = _site_group(p["properties"].get("site")
                            or p["properties"].get("city") or "")
            if s and s not in day_sites:
                day_sites.append(s)
        lines.append(f"\nDay {i} - {', '.join(day_sites)}")
        for p in sorted(day_places, key=lambda p: (slot_of(p), p["properties"]["name"])):
            pr = p["properties"]
            e = g.sources(p["id"])[0]
            ep = e["properties"]
            reel = g.reel_node(e)
            rp = reel.get("properties", {})
            lines.append(f"  [{slot_names[slot_of(p)]}] {pr['name']} ({pr.get('site') or ''})")
            if ep.get("recommendation"):
                lines.append(f"    {ep['recommendation']}")
            title = rp.get("title") or ep.get("reel_title") or "Reel"
            creator = rp.get("creator") or ep.get("creator") or ""
            lines.append(f'    Source: "{title}" by {creator} - {rp.get("url") or ""}')
    if overflow:
        lines.append(f"\nNot fitted into {days} day(s) at {MAX_STOPS_PER_DAY} stops/day "
                     f"({len(overflow)} more place(s) in scope - add a day or ask for a "
                     f"longer itinerary):")
        for p in overflow:
            pr = p["properties"]
            e = g.sources(p["id"])[0]
            reel = g.reel_node(e)
            rp = reel.get("properties", {})
            title = rp.get("title") or e["properties"].get("reel_title") or "Reel"
            creator = rp.get("creator") or e["properties"].get("creator") or ""
            lines.append(f"  - {pr['name']} ({pr.get('site') or ''})")
            lines.append(f'    Source: "{title}" by {creator} - {rp.get("url") or ""}')
    if len(plan) < days:
        lines.append(f"\n(Only {len(stream)} place(s) are in the graph for this scope, "
                     f"so the plan uses {len(plan)} day(s) rather than {days}.)")
    return "\n".join(lines)


def vocab(g):
    names = lambda t: [n["properties"]["name"] for n in g.by_type.get(t, [])]
    return {"themes": names("Theme"), "states": names("State"),
            "regions": names("Region"), "countries": names("Country"),
            "sites": names("Site"),
            "creators": [n["properties"]["handle"] for n in g.by_type.get("Creator", [])]}


def canonical_question(parsed):
    """Turn a validated LLM intent dict into a canonical question string
    that the rule pipeline already knows how to answer. Facts still come
    only from the graph — this is just a rewrite of the question."""
    if not parsed:
        return None
    intent = parsed["intent"]
    if intent == "between" and parsed.get("site_a") and parsed.get("site_b"):
        return f"places between {parsed['site_a']} || {parsed['site_b']}"
    scope = parsed.get("site") or parsed.get("geo") or ""
    if intent == "count":
        return "how many places do we have?"
    if intent == "reels":
        return "list reels"
    if intent == "creator" and parsed.get("creator"):
        return f"what did {parsed['creator']} recommend?" + (f" in {scope}" if scope else "")
    if intent == "theme" and parsed.get("theme"):
        return f"best {parsed['theme']} spots" + (f" in {scope}" if scope else "")
    if scope and intent in ("browse", "theme", "creator", "unknown"):
        # A named scope with a missing primary slot (e.g. "Mammoth Cave
        # highlights" parsed as theme-with-site but no theme) still means:
        # show what's in that place.
        return f"places in {scope}"
    if intent == "browse" and scope:
        return f"places in {scope}"
    if intent == "itinerary":
        days = parsed.get("days") or 2
        return f"make a {days}-day itinerary" + (f" for {scope}" if scope else "")
    return None


def route_rules(g, question, raw_question=None):
    """Canonical question drives the rules; raw_question (the user's own
    words, when the canonical form came from the LLM parse) is what the
    itinerary builder reads for intent signals the parse drops — "NPs",
    or whether a trip length was actually given instead of defaulted."""
    q = " " + re.sub(r"\s+", " ", question.lower()).strip() + " "
    raw_q = raw_question if raw_question is not None else question
    result = None
    m = re.search(r"places between (.+?) \|\| (.+)", q)
    if m:
        result = answer_between(g, m.group(1).strip(), m.group(2).strip())
    if re.search(r"how many|count|total", q):
        result = answer_count(g)
    elif re.search(r"itinerar|road trip|\d+\s*[- ]?\s*days?\b|weekend|plan (a|my|the)", q):
        result = answer_itinerary(g, q, raw_q)
    elif "reel" in q and ("list" in q or "what" in q or "show" in q or "all" in q):
        result = answer_reels(g)
    elif "@" in raw_q:
        result = answer_creator(g, q, raw_q)
    if result is None:
        result = answer_theme(g, q, raw_q)
    if result is None:
        result = answer_browse(g, q, raw_q)
    if result is None and ("recommend" in q or "by " in q):
        result = answer_creator(g, q, raw_q)
    if result is None:
        result = ("I can only answer from the Reel Places graph. Try things like:\n"
                  '- "best sunrise spots" or "best waterfall spots"\n'
                  '- "what did @amorywild recommend?"\n'
                  '- "what\'s recommended in Zion National Park?"\n'
                  '- "places in California" / "places in the Southwest"\n'
                  '- "list reels" / "how many places do we have?"\n'
                  '- "make a 2-day itinerary for Utah"')
    return result


_SITE_SUFFIXES = (" National Park and Preserve", " National Park", " State Park",
                  " National Monument", " National Recreation Area",
                  " National Conservation Area", " National Historical Park")


def mention_rescue(g, question):
    """Last-resort grounding when the LLM's parse yields no canonical
    question: if the question verbatim names a graph vocabulary entry (a
    site, a state/region/country, or a creator handle), use that scope.
    This invents nothing — the mention must appear in the user's own words
    and in the graph's vocabulary, and the answer still comes entirely
    from the graph pipeline. Themes are deliberately NOT rescued: loose
    theme guessing is what made the old rule-based mode wrong."""
    q = " " + question.lower() + " "
    v = vocab(g)
    if "between" in q:
        # A between-question must never rescue to a single endpoint's
        # highlights (2026-11-11: that is exactly the wrong answer).
        mentioned = []
        for site in v["sites"]:
            forms = [site] + [site[: -len(s)] for s in _SITE_SUFFIXES if site.endswith(s)]
            for form in forms:
                if len(form) >= 5 and form.lower() in q:
                    mentioned.append((q.index(form.lower()), site))
                    break
        mentioned.sort()
        seen = []
        for _, site in mentioned:
            if site not in seen:
                seen.append(site)
        if len(seen) >= 2:
            return f"places between {seen[0]} || {seen[1]}"
    best = None  # (length, canon)
    for site in v["sites"]:
        forms = [site] + [site[: -len(s)] for s in _SITE_SUFFIXES if site.endswith(s)]
        for form in forms:
            if len(form) >= 5 and form.lower() in q:
                if best is None or len(form) > best[0]:
                    best = (len(form), f"places in {site}")
    if best:
        return best[1]
    for geo in v["states"] + v["regions"] + v["countries"]:
        if geo.lower() in q:
            return f"places in {geo}"
    for creator in v["creators"]:
        if creator.lower() in q or creator.lower().lstrip("@") in q:
            return f"what did {creator} recommend?"
    return None


def answer_places(g, grounded):
    """The places a grounded answer recommends, with coordinates — parsed
    from the answer's own place headers (fmt_place lines and itinerary
    stop lines) and matched back to Place nodes, so the UI can map
    exactly the places the answer names (user, 2026-11-11: answers
    should include a map of the places recommended). Returns a list of
    {"name", "site", "city", "state", "lat", "lng"} in answer order."""
    by_name = {}
    for p in g.by_type.get("Place", []):
        by_name.setdefault(p["properties"]["name"], []).append(p)
    found, seen = [], set()

    def add(node):
        pr = node["properties"]
        if node["id"] in seen or pr.get("lat") is None or pr.get("lng") is None:
            return
        seen.add(node["id"])
        found.append({"name": pr["name"], "site": pr.get("site") or "",
                      "city": pr.get("city") or "", "state": pr.get("state") or "",
                      "lat": pr["lat"], "lng": pr["lng"]})

    def pick(name, state=None, site=None):
        cands = by_name.get(name, [])
        if state:
            narrowed = [p for p in cands if p["properties"].get("state") == state]
            if narrowed:
                cands = narrowed
        if site and len(cands) > 1:
            narrowed = [p for p in cands if p["properties"].get("site") == site]
            if narrowed:
                cands = narrowed
        if cands:
            add(cands[0])

    for ln in grounded.split("\n"):
        m = re.match(r"^- (.+?) - .* \([^,]+, ([^,]+), [^)]+\)", ln)  # fmt_place header
        if m:
            pick(m.group(1), state=m.group(2).strip())
            continue
        m = re.match(r"^\s*\[[^\]]+\] (.+?) \(([^)]*)\)", ln)  # itinerary stop
        if m:
            pick(m.group(1), site=m.group(2).strip())
            continue
        m = re.match(r"^\s+- (.+?) \(([^)]+)\)\s*$", ln)  # itinerary overflow line
        if m:
            pick(m.group(1), site=m.group(2).strip())
    return found


def build_references(grounded):
    """Deterministic References section for a grounded answer (user
    format, 2026-11-11: summary first, then references). Parses the
    grounded text's own place headers and Source lines, so every source
    the graph produced is listed exactly as written — nothing dropped,
    nothing invented, no LLM involved. Returns "" when the answer has
    no sources (e.g. the count answer)."""
    refs, seen = [], set()
    current = None
    lines = grounded.split("\n")
    for i, ln in enumerate(lines):
        s = ln.strip()
        m = re.match(r"^- (.+?) - ", ln)  # fmt_place header
        if m:
            current = m.group(1)
            continue
        m = re.match(r"^\s*\[[^\]]+\] (.+?) \(", ln)  # itinerary stop line
        if m:
            current = m.group(1)
            continue
        m = re.match(r"^\s+- (.+?) \(([^)]+)\)\s*$", ln)  # itinerary overflow line
        if m:
            current = m.group(1)
            continue
        m = re.match(r'^Source: "(.+)" by (.+?) - (https?://\S+)$', s)
        if m:
            entry = (current, f'"{m.group(1)}" by {m.group(2)} - {m.group(3)}')
            if entry not in seen:
                seen.add(entry)
                refs.append(entry)
            continue
        if re.fullmatch(r"https?://\S+", s) and i > 0:
            prev = lines[i - 1].strip()  # reels list: URL under its title line
            if prev.startswith('- "'):
                entry = (None, prev[2:] + " - " + s)
                if entry not in seen:
                    seen.add(entry)
                    refs.append(entry)
    if not refs:
        return ""
    return "\n".join(f"- {place} — {src}" if place else f"- {src}"
                     for place, src in refs)


def _answer_full(g, question, backend_note):
    """LLM-first mode (user directives, 2026-11-02 — the rule-only mode
    gave wrong answers on the website, so there is NO rule-based fallback
    for facts). The LLM parses the question into a validated intent drawn
    from the graph's vocabulary; the graph pipeline produces the facts,
    and the answer is presented as a short LLM summary plus a
    deterministic References section (user, 2026-11-11). Returns
    {"text": ..., "places": [...]} — places are the graph places the
    answer recommends (with coordinates, for the UI's map); web and
    error answers carry no places. When the graph has no answer, the bot
    falls back to a labelled web search (user directive, 2026-11-02)
    instead of dead-ending. If the AI connection itself is down, the
    bot says so."""
    llm = None
    try:
        import llm as _llm
        llm = _llm if _llm.available() else None
    except Exception:
        llm = None
    if llm is None:
        return {"text": ("This chatbot runs in LLM-only mode, but the AI connection "
                         "isn't available right now, so I can't answer that."),
                "places": []}
    def web_fallback():
        """User directive (2026-11-02): no dead ends — when the graph
        can't answer, search the web and answer from it, clearly labelled
        as web-sourced (never presented as graph facts)."""
        web = llm.web_answer(question)
        if not web:
            return None
        return ("🌐 **From the web** — the Reel Places graph doesn't cover "
                "this yet, so I searched the web:\n\n" + web)

    parsed = llm.parse_question(question, vocab(g))
    canon = canonical_question(parsed)
    if (canon and parsed and parsed.get("intent") in ("itinerary", "theme")
            and not (parsed.get("site") or parsed.get("geo"))):
        # The LLM sometimes drops the scope (state/site) from the parse —
        # recover it deterministically from the user's own words, so a
        # "2-day itinerary for Utah" never silently becomes an all-places
        # plan (found 2026-11-11: 1 in 4 parses dropped geo=Utah).
        qlow = " " + re.sub(r"\s+", " ", question.lower()).strip() + " "
        sites = find_sites(g, qlow)
        scope = sites[0]["properties"]["name"] if sites else None
        if not scope:
            kind, node = find_geo(g, qlow, question)
            scope = node["properties"]["name"] if node else None
        if scope:
            if parsed["intent"] == "itinerary" and " for " not in canon:
                canon = canon + f" for {scope}"
            elif parsed["intent"] == "theme" and scope.lower() not in canon.lower():
                canon = canon + f" in {scope}"
    rescued = False
    if not canon:
        canon = mention_rescue(g, question)
        rescued = canon is not None
    if not canon:
        err = getattr(llm, "LAST_ERROR", None)
        if err:
            return {"text": ("The AI call itself failed, so I couldn't parse that "
                             f"question. Reason: {err}. Check the API key on the page "
                             "and try again."), "places": []}
        web = web_fallback()
        if web:
            return {"text": web, "places": []}
        return {"text": ("I couldn't understand that question well enough to query "
                         "the graph, and the web fallback didn't return an answer "
                         "either. Try naming a theme (sunrise, waterfall, hike), "
                         "a place or park, a creator like @amorywild, or ask for an "
                         "itinerary with a number of days."), "places": []}
    result = route_rules(g, canon, question)
    places = []
    if result.startswith("I can only answer") or result.startswith("I don't have "):
        web = web_fallback()
        if web:
            return {"text": web, "places": []}
    if not result.startswith("I can only answer") and not result.startswith("I don't have "):
        # The places this answer recommends, for the UI's map — parsed
        # from the grounded answer itself, before summarising.
        places = answer_places(g, result)
        # Summary + references (user, 2026-11-11: full answers were too
        # long). The LLM writes only the summary (no URLs allowed in it);
        # the References section is built deterministically from the
        # grounded answer, so sources can't drift. No-match replies are
        # already short and stay as they are.
        summary = llm.phrase_answer(question, result)
        if summary:
            refs = build_references(result)
            result = summary + (f"\n\nReferences:\n{refs}" if refs else "")
    # No technical footer on answers (user, 2026-11-02: "I don't care for
    # that") — answers end with the answer itself.
    return {"text": result, "places": places}


@tracing.track("reel-chat-answer")
def _traced_answer(g, question, backend_note):
    """Traced core — flushes pending Opik traces before returning."""
    try:
        return _answer_full(g, question, backend_note)
    finally:
        tracing.flush()


def answer(g, question, backend_note):
    """Public text-only entry point (CLI + backwards compatibility)."""
    return _traced_answer(g, question, backend_note)["text"]


def answer_full(g, question, backend_note):
    """Public entry point for the UI: {"text", "places"} (places carry
    coordinates so the app can map the recommended places)."""
    return _traced_answer(g, question, backend_note)


def main():
    try:
        from signal import SIG_DFL, SIGPIPE, signal
        signal(SIGPIPE, SIG_DFL)  # no BrokenPipeError noise when piped to head
    except Exception:
        pass
    data, backend_note = load_graph()
    g = Graph(data)
    if len(sys.argv) > 1:
        print(answer(g, " ".join(sys.argv[1:]), backend_note))
        return
    print(f"Reel Places chatbot - answering only from the {backend_note}.")
    print("Ask about places, themes, creators, reels, or an itinerary. Ctrl-D / 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            print()
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            print(answer(g, question, backend_note))


if __name__ == "__main__":
    main()
