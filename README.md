# Reel Places

**Every place worth visiting that hides inside Instagram travel reels — extracted, verified, geocoded, and organized on one page with a map and a chatbot.**

Send a reel, and Reel Places pulls out every location it names, checks each one against a real map, records *what the reel recommended it for* (the sunrise, the hike, the viewpoint…), and files it under **Region → Park or City → State → Activity → Trail or Site**. Duplicates never get a second pin — a second reel recommending the same place is added as another source on the same entry, so the best-covered places rise to the top.

## Live

- 🌐 **Website:** https://harikrishnad1997.github.io/reel-places/
- 🗺️ **Map:** https://harikrishnad1997.github.io/reel-places/map.html
- 💬 **Reel Chat (the project's one chatbot):** https://harikrishnad1997.github.io/reel-places/chat.html — takes you straight to the Streamlit app
- 🤖 **Reel Chat on Streamlit:** https://reel-places-5qsqnj7f8465py9mdhpttg.streamlit.app/
- 📦 **Repository:** https://github.com/harikrishnad1997/reel-places

## Features

- **Region-first browsing** — places grouped Region → National/State Park or City → State → Activity → Trail/Site, so "sunrise spots in Zion" is a scroll, not a search
- **One pin per place, every source attached** — each entry and map popup carries every recommending reel: creator, reel title, rank in the reel, its "best for" tag, and the recommendation in the creator's own framing
- **Verified coordinates** — every pin is geocoded against OpenStreetMap and validated inside its state (and park, where relevant); same-name traps (a gift shop called "The Beehive", a bus stop called "Horseshoe Falls") are rejected, and anything that can't be pinned honestly is left out rather than guessed
- **A graph, not a spreadsheet** — the data also builds a Kuzu knowledge graph (Place · Site · State · Region · Country · Reel · Creator · Theme), which powers Reel Chat's answers and itinerary builder
- **LLM-only chatbot** — Reel Chat parses questions with an LLM into graph-checked intents, answers only from graph facts, and rephrases with link-preservation checks; if the AI call fails it says why instead of guessing

## How it works

1. A reel link arrives in chat with Muse.
2. Muse reads the caption **and** the video itself (media understanding), extracting every named location and what it was recommended for.
3. Each spot is verified and geocoded; duplicates merge as additional sources.
4. Content rules are applied (below), then `places.json` / `reels.json` are updated — the single source of truth.
5. `build_site.py` regenerates the list page and the Leaflet map; the graph and the Streamlit app's data bundle are rebuilt; everything is pushed to GitHub, and GitHub Pages + Streamlit Cloud redeploy.

### Content rules

- **USA only** — US territories included (American Samoa, US Virgin Islands stay); foreign places are excluded.
- **No magic tricks** — reels where a park is only a backdrop for a card trick recommend nothing, so they're excluded; a place whose only sources are magic reels is removed, and can return when a real recommendation appears.
- **No invented spots** — unnamed waterfalls and trails stay at park level; a spot is never pinned to a namesake's coordinates.

## Data model

`places.json` — one object per place:

| Field | Meaning |
|---|---|
| `name` | Trail / site / place name as the reel stated it |
| `site` | The park or named area it belongs to (grouping level 2) |
| `city`, `state`, `region`, `country` | Location and the page's region grouping |
| `lat`, `lng` | Verified coordinates |
| `best_for` | The activity it was recommended for (Sunrise, Hike, Waterfall, View…) |
| `recommendation` | What the reel said about it |
| `source_reel`, `source_title`, `source_creator`, `source_rank` | The primary recommending reel |
| `additional_sources[]` | Every later reel that recommended the same place |

`reels.json` — one object per processed reel: title, creator, URL, theme, recommendation summary, and how many places it produced.

## Reel Chat

Reel Chat is a Streamlit app over the Kuzu graph (with a `graph.json` snapshot as its hosted data source). It answers questions like *"Best sunrise spots in Utah"*, *"What did @amorywild recommend in Zion?"*, and *"Make a 2-day itinerary for Utah"* — grouping stops by park, ordering them nearest-neighbour, and slotting sunrise spots in the morning and sunset spots in the evening. It is LLM-only by design: the model parses and phrases, the graph decides what's true. On Streamlit Cloud you paste your own OpenAI API key into the page (kept in your browser session only, never stored), or set it once as an app secret.

## Tech stack

- **Site:** Jekyll (GitHub Pages, cayman theme) + Leaflet / OpenStreetMap
- **Data:** JSON source of truth, OpenStreetMap Nominatim geocoding
- **Graph:** Kuzu embedded graph database, generated from the JSON data
- **Chatbot:** Streamlit + OpenAI (parse/phrase only; facts graph-verified)
- **Pipeline:** Muse (extraction, media understanding, geocoding, build, deploy)

## Repository layout

```
index.md       the places list (generated)
map.html       the Leaflet map (generated)
chat.html      Reel Chat — redirects to the Streamlit app (the one chatbot)
places.json    places data (mirrors the pipeline source of truth)
reels.json     reels data
app/           Streamlit app bundle (streamlit_app.py, chatbot.py, llm.py, graph.json)
LICENSE        MIT
```

The generator and pipeline live outside this repo: `places.json` is edited there and copied in by the build, so this repo always reflects the published state.

## Running locally

```bash
# regenerate the site from the data
python3 build_site.py            # writes index.md + map.html into the staging folder

# rebuild the graph + chat data
cd graph && .venv/bin/python build_graph.py

# run Reel Chat locally (uses the secure OpenAI connector — no key handling)
cd graph && .venv/bin/streamlit run streamlit_app.py
```

## Project status

- **711 places · 588 source reels · 10 regions** and counting
- Built initially from @amorywild's national-park reels: 237 of his posts processed in batches (his older posts aren't reachable through Instagram's profile listing — sending a specific reel always works)

## Contributing

Found a reel with a great spot? Send the link to the project owner in chat with Muse — the pipeline handles extraction, verification, and publishing, and the place appears on the page and map with its source attached.

## License

MIT — see [LICENSE](LICENSE). The MIT License covers the code and site content in this repository; the linked Instagram reels belong to their creators.
