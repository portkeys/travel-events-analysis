# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analytics project for Outside Travel's embed widget events. Data flows from RudderStack → S3 (gzipped JSON) → local Parquet cache → analysis/dashboards. The project tracks user engagement with a travel booking widget embedded on Outside's sister sites (BikeReg, SkiReg, RunReg, TriReg, SKI Magazine).

## Key Commands

```bash
# Install dependencies (uses uv)
uv sync

# Fetch event data from S3 into local Parquet cache
python -m travel_events load --days 30
python -m travel_events load --start 2025-02-15 --end 2025-03-17

# Run the interactive Streamlit dashboard
streamlit run travel_events/dashboard.py

# Generate static HTML report (outputs to docs/index.html)
python generate_html.py
```

The S3 loader requires AWS credentials via the `wen_outside` profile (configurable with `--profile`).

## Architecture

**Data pipeline:** `loader.py` fetches gzipped JSON from S3 (`outside-rudderstack-prod` bucket), normalizes RudderStack events into flat DataFrames via `_normalize_events()`, and caches as daily Parquet files in `data/cache/`. Today's date is always re-fetched; other days use cache. S3 downloads are parallelized (10 files/day, 4 days concurrently).

**Analysis layer:** `analysis.py` provides reusable aggregation functions (funnel, UTM breakdown, engaged users, destinations, referrer breakdown) consumed by both the Streamlit dashboard and the static HTML generator.

**Two output paths exist for the same data:**
- `travel_events/dashboard.py` — interactive Streamlit app with date range sidebar, reads from Parquet cache (no S3 calls)
- `generate_html.py` — generates a self-contained static HTML page at `docs/index.html` using the same analysis functions and Plotly charts

Both outputs apply identical data transforms: stripping "None" strings from `user_id`, removing internal users (`@outsideinc.com`, `@tenex.co`), and inferring `utm_source` from `referring_domain` when missing.

**Geocoding:** Search destinations and UTM term locations are geocoded via Nominatim with a persistent JSON cache at `data/geocache.json` (1 req/sec rate limit).

## Brand Styling

Outside brand palette: signature yellow `#FFD100`, black, and grays. The `COLORS` dict is defined in both `dashboard.py` and `generate_html.py`. Use the `/outside-brand-style` skill when creating visualizations.

## Data Notes

- The `data/` directory is gitignored — Parquet cache and CSVs must be generated locally via the `load` command
- Event funnel steps: Loaded a Page → Embed Widget Viewed → Embed Widget Clicked → Travel Searched → Property Added to Cart → Checkout Clicked
- Sister site referrer domains are normalized to short names (e.g., `www.bikereg.com` → `BikeReg`) in both `analysis.py` and `dashboard.py`
