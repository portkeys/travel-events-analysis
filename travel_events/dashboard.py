"""Streamlit dashboard for travel widget event analysis."""

import io
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure the project root is on sys.path so imports work when running via
# `streamlit run travel_events/dashboard.py` from the repo root.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import re
import time

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from geopy.geocoders import Nominatim

from travel_events.analysis import (
    daily_event_counts,
    engaged_users,
    event_summary,
    funnel,
    top_search_destinations,
    utm_breakdown,
)
from travel_events.loader import CACHE_DIR, load_date_range

# --- Outside brand palette: yellow (#FFD100), black, grays ---
COLORS = {
    "primary": "#FFD100",       # Outside signature yellow
    "primary_dark": "#E6BC00",  # dark gold
    "primary_light": "#FFF3B0", # pale yellow
    "black": "#000000",
    "dark_grey": "#333333",
    "mid_grey": "#666666",
    "grey": "#999999",
    "light_grey": "#CCCCCC",
    "bg_grey": "#F7F7F7",
    "accent": "#FFD100",        # use yellow as the accent/highlight
    "secondary": "#333333",     # dark grey for secondary data
}

st.set_page_config(page_title="Outside Travel Insights", layout="wide")

# Custom CSS for styled KPI cards and Outside branding
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-top: 3px solid #FFD100;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    [data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #666666 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Outside Travel Insights")


@st.cache_data(ttl=300)
def _load_cached_range(start: date, end: date):
    """Load from local Parquet cache only — no S3 calls."""
    frames = []
    current = start
    while current <= end:
        path = CACHE_DIR / f"{current.isoformat()}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
        current += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_GEOCACHE_PATH = Path(_PROJECT_ROOT) / "data" / "geocache.json"


@st.cache_data(ttl=86400)  # cache geocoding results for 24 hours in Streamlit
def _geocode_locations(locations: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    """Geocode a list of location strings, using a persistent JSON cache on disk."""
    # Load disk cache
    disk_cache: dict[str, tuple[float, float] | None] = {}
    if _GEOCACHE_PATH.exists():
        try:
            disk_cache = json.loads(_GEOCACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    geolocator = Nominatim(user_agent="travel-events-analysis", timeout=5)
    results: dict[str, tuple[float, float]] = {}
    new_lookups = 0

    for loc in locations:
        if loc in disk_cache:
            if disk_cache[loc] is not None:
                results[loc] = tuple(disk_cache[loc])
            continue
        # New location — geocode it
        # Add ", USA" hint for bare city names (no comma = no state/country)
        query = loc if "," in loc else f"{loc}, USA"
        try:
            if new_lookups > 0:
                time.sleep(1.1)  # respect Nominatim 1 req/sec rate limit
            geo = geolocator.geocode(query)
            new_lookups += 1
            if geo:
                coords = (geo.latitude, geo.longitude)
                results[loc] = coords
                disk_cache[loc] = coords
            else:
                disk_cache[loc] = None
        except Exception:
            disk_cache[loc] = None

    # Persist new results to disk
    if new_lookups > 0:
        _GEOCACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _GEOCACHE_PATH.write_text(json.dumps(disk_cache, indent=2))

    return results


# --- Sidebar: date range controls ---
st.sidebar.header("Date Range")
today = date.today()
default_start = today - timedelta(days=30)

start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=today)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

df = _load_cached_range(start_date, end_date)

# Guard against "None" strings from Parquet cache coercion
if "user_id" in df.columns:
    df.loc[df["user_id"].isin(["None", "none", ""]), "user_id"] = None
    # Remove internal users (Outside Inc. and Tenex employees)
    internal_mask = df["user_id"].str.contains(r"@outsideinc\.com|@tenex\.co", na=False)
    df.loc[internal_mask, "user_id"] = None

# Fill missing utm_source from referring_domain so direct/travel-site traffic
# appears in UTM charts. Without this, all events with utm_source=None (including
# all searches, cart adds, and checkouts) are silently dropped from UTM analysis.
_REFERRER_TO_SOURCE = {
    "www.skimag.com": "SKI",
    "cdn.skimag.com": "SKI",
    "www.bikereg.com": "BikeReg",
    "www.runreg.com": "RunReg",
    "www.skireg.com": "SkiReg",
    "www.trireg.com": "TriReg",
    "www.outsideonline.com": "Outside",
    "travel.outsideonline.com": "(direct / travel site)",
}
if "utm_source" in df.columns:
    missing_utm = df["utm_source"].isna()
    if missing_utm.any():
        inferred = df.loc[missing_utm, "referring_domain"].map(_REFERRER_TO_SOURCE)
        df.loc[missing_utm, "utm_source"] = inferred.fillna("(direct / travel site)")
    # Also fill missing utm_campaign — utm_breakdown groups by (source, campaign),
    # and groupby drops rows where campaign is NaN
    if "utm_campaign" in df.columns:
        df["utm_campaign"] = df["utm_campaign"].fillna("(none)")

if df.empty:
    st.warning(
        "No cached data for this date range. "
        "Run `python -m travel_events load --days 30` to fetch data from S3 first."
    )
    st.stop()

st.caption(f"Showing data from **{start_date}** to **{end_date}** ({len(df):,} events)")


# ============================================================
# 1. Audience Overview — who are our users?
# ============================================================
st.header("Audience Overview")

# Compute user segments
total_unique = df["anonymous_id"].nunique()
registered_count = df["user_id"].dropna().nunique()
ids_with_uid = set(df[df["user_id"].notna()]["anonymous_id"].unique())
anon_df = df[~df["anonymous_id"].isin(ids_with_uid) & df["user_id"].isna()]
anon_count = anon_df["anonymous_id"].nunique()
anon_sessions = anon_df.groupby("anonymous_id")["session_id"].nunique()
single_visit_anon = int((anon_sessions == 1).sum())
repeat_anon = int((anon_sessions > 1).sum())
repeat_pct = repeat_anon / anon_count * 100 if anon_count > 0 else 0

# Row 1: top-level KPIs (4 cols for alignment)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", f"{len(df):,}")
col2.metric("Unique Event Types", f"{df['event'].nunique()}")
col3.metric("Unique Users", f"{total_unique:,}")
col4.metric("Registered Users", f"{registered_count:,}")

# Row 2: anonymous user breakdown (4 cols, aligned)
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Anonymous Users", f"{anon_count:,}")
col_b.metric("Single-Visit Anon.", f"{single_visit_anon:,}")
col_c.metric("Repeat Anonymous", f"{repeat_anon:,}")
col_d.metric("Repeat Rate", f"{repeat_pct:.1f}%")

# Visual breakdown: stacked bar showing user composition
st.subheader("User Composition")
if repeat_anon > 0:
    st.markdown(
        f"**{repeat_anon:,} repeat anonymous users** visited multiple times without registering "
        f"— these represent your best activation opportunity."
    )

reg_pct = registered_count / (registered_count + repeat_anon + single_visit_anon) * 100
# Custom text per slice: show label+percent for the two visible slices, blank for Registered
single_pct = single_visit_anon / (registered_count + repeat_anon + single_visit_anon) * 100
repeat_pct_pie = repeat_anon / (registered_count + repeat_anon + single_visit_anon) * 100
fig_comp = go.Figure(go.Pie(
    labels=["Single-Visit Anonymous", "Repeat Anonymous", "Registered"],
    values=[single_visit_anon, repeat_anon, registered_count],
    marker=dict(
        colors=[COLORS["light_grey"], COLORS["dark_grey"], COLORS["primary"]],
        line=dict(color="white", width=2),
    ),
    textinfo="text",
    text=[
        f"Single-Visit Anonymous<br>{single_pct:.1f}%",
        f"Repeat Anonymous<br>{repeat_pct_pie:.1f}%",
        "",  # hide — the annotation callout handles this
    ],
    textposition="outside",
    textfont=dict(size=14, color="#000000"),
    pull=[0, 0.04, 0.15],
    sort=False,
    direction="clockwise",
))
# Annotation with arrow pointing to the registered slice.
# The registered slice is tiny and pulled out, sitting at the boundary between
# the two larger slices. In Plotly pie coords, the pie center is (0.5, 0.5).
# With clockwise direction and no sort, slices go: Single-Visit (large grey),
# Repeat Anonymous (small dark), Registered (tiny yellow, pulled out).
# The registered slice sits roughly at the 12 o'clock position (top of pie).
fig_comp.add_annotation(
    text=f"<b>Registered: {registered_count:,} ({reg_pct:.2f}%)</b>",
    x=0.52, y=1.0,  # top of pie where the tiny registered slice is
    showarrow=True,
    arrowhead=2,
    arrowwidth=2,
    arrowcolor=COLORS["black"],
    ax=80, ay=30,  # label box offset to upper-right
    font=dict(size=13, color=COLORS["black"]),
    bgcolor=COLORS["primary"],
    bordercolor=COLORS["black"],
    borderwidth=1,
    borderpad=6,
)
fig_comp.update_layout(
    height=400,
    margin=dict(l=40, r=40, t=30, b=30),
    showlegend=False,
)
st.plotly_chart(fig_comp, use_container_width=True)


# ============================================================
# 2. Event Summary — what's happening?
# ============================================================
st.header("Event Summary")

summary = event_summary(df)
summary_sorted = summary.sort_values("count", ascending=True)

fig_events = go.Figure()
fig_events.add_trace(go.Bar(
    y=summary_sorted.index,
    x=summary_sorted["count"],
    orientation="h",
    marker_color=[
        COLORS["primary"] if name in ("Travel Searched", "Embed Widget Clicked",
                                       "Property Added to Cart", "Checkout Clicked")
        else COLORS["dark_grey"]
        for name in summary_sorted.index
    ],
    text=[f"{v:,}" for v in summary_sorted["count"]],
    textposition="outside",
))
fig_events.update_layout(
    title="Page loads dominate — engagement events are <1% of traffic",
    height=max(300, len(summary_sorted) * 40 + 80),
    margin=dict(l=0, r=80, t=40, b=0),
    xaxis_title="Event Count",
    yaxis_title="",
    showlegend=False,
)
st.plotly_chart(fig_events, use_container_width=True)

with st.expander("Event breakdown table"):
    st.dataframe(summary, use_container_width=True)


# ============================================================
# 3. Daily Trend — how are events changing over time?
# ============================================================
st.header("Daily Event Trend")

daily = daily_event_counts(df)

# Split into high-volume (Page Loads, Widget Views) and low-volume (everything else)
high_vol_cols = [c for c in ["Loaded a Page", "Embed Widget Viewed"] if c in daily.columns]
low_vol_cols = [c for c in daily.columns if c not in high_vol_cols and daily[c].sum() > 0]

# High-volume: Page Loads vs Widget Views
if high_vol_cols:
    st.subheader("Page Loads vs Widget Views")
    fig_high = go.Figure()
    high_colors = [COLORS["light_grey"], COLORS["primary"]]
    for i, col in enumerate(high_vol_cols):
        fig_high.add_trace(go.Scatter(
            x=daily.index, y=daily[col],
            name=col, mode="lines",
            line=dict(color=high_colors[i], width=2),
        ))
    fig_high.update_layout(
        height=350, hovermode="x unified",
        yaxis_title="Events", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_high, use_container_width=True)



# ============================================================
# 4. Conversion Funnel
# ============================================================
st.header("Conversion Funnel")
funnel_df = funnel(df)
if not funnel_df.empty:
    st.markdown("Separated by user type so registered users are visible on their own scale.")

    col_funnel_anon, col_funnel_reg = st.columns(2)

    with col_funnel_anon:
        fig_funnel_anon = go.Figure(go.Funnel(
            y=funnel_df["step"],
            x=funnel_df["anonymous_users"],
            textinfo="value+percent initial",
            texttemplate="%{value:,} (%{percentInitial})",
            marker_color=COLORS["dark_grey"],
            textfont=dict(size=12),
        ))
        fig_funnel_anon.update_layout(
            title="Anonymous Users",
            height=420,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_funnel_anon, use_container_width=True)

    with col_funnel_reg:
        fig_funnel_reg = go.Figure(go.Funnel(
            y=funnel_df["step"],
            x=funnel_df["registered_users"],
            textinfo="value+percent initial",
            texttemplate="%{value:,} (%{percentInitial})",
            marker_color=COLORS["primary"],
            textfont=dict(size=12),
        ))
        fig_funnel_reg.update_layout(
            title="Registered Users",
            height=420,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_funnel_reg, use_container_width=True)

    with st.expander("Funnel data table"):
        st.dataframe(funnel_df, use_container_width=True)


# ============================================================
# 5. UTM Performance
# ============================================================
st.header("UTM Performance")
utm_df = utm_breakdown(df)

if not utm_df.empty:
    utm_flat = utm_df.reset_index()

    by_source = (
        utm_flat.groupby("utm_source")[["impressions", "widget_views", "clicks", "searches"]]
        .sum()
        .sort_values("impressions", ascending=False)
        .head(15)
    )

    # --- Volume charts: impressions/views and widget clicks ---
    # Searches excluded from this chart — they only happen on the direct travel site
    # (not on sister sites), so comparing them side-by-side with clicks is misleading.
    st.subheader("Traffic Volume by UTM Source")
    st.markdown("Page loads & widget views (left) vs. widget clicks (right) — separate scales so low-volume clicks are visible.")

    col_vol, col_eng = st.columns(2)

    by_source_asc = by_source.sort_values("impressions", ascending=True)
    fixed_y_order = by_source_asc.index.tolist()

    with col_vol:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            y=by_source_asc.index, x=by_source_asc["impressions"],
            name="Page Loads", orientation="h",
            marker_color=COLORS["light_grey"],
            text=[f"{v:,}" for v in by_source_asc["impressions"]],
            textposition="outside",
        ))
        fig_vol.add_trace(go.Bar(
            y=by_source_asc.index, x=by_source_asc["widget_views"],
            name="Widget Views", orientation="h",
            marker_color=COLORS["primary"],
            text=[f"{v:,}" for v in by_source_asc["widget_views"]],
            textposition="outside",
        ))
        fig_vol.update_layout(
            title=dict(text="Page Loads & Widget Views", y=0.98),
            barmode="group", height=480,
            yaxis={"categoryorder": "array", "categoryarray": fixed_y_order},
            xaxis_title="Count",
            margin=dict(l=0, r=60, t=70, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_eng:
        fig_eng = go.Figure()
        fig_eng.add_trace(go.Bar(
            y=by_source_asc.index, x=by_source_asc["clicks"],
            name="Widget Clicks", orientation="h",
            marker_color=COLORS["primary"],
            text=[f"{v:,}" for v in by_source_asc["clicks"]],
            textposition="outside",
        ))
        fig_eng.update_layout(
            title=dict(text="Widget Clicks", y=0.98),
            height=480,
            yaxis={"categoryorder": "array", "categoryarray": fixed_y_order},
            xaxis_title="Count",
            margin=dict(l=0, r=60, t=70, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_eng, use_container_width=True)

    # --- Conversion rates: ranked bar charts per metric ---
    # Each rate uses the correct denominator based on the funnel:
    #   View Rate = widget views / page loads (did users scroll to the widget?)
    #   Click Rate = clicks / widget views (of those who saw it, who clicked?)
    #   Search Rate = searches / page loads (search is a separate funnel from widget clicks)
    st.subheader("Conversion Rates by UTM Source (Ranked)")
    st.markdown(
        "**View Rate** = widget views / page loads (did users scroll to the widget?). "
        "**Click Rate** = clicks / widget views (of those who saw it, who clicked?). "
        "**Search Rate** = searches / page loads (search is a separate action from the widget)."
    )
    rates = by_source.copy()
    rates["View Rate %"] = (rates["widget_views"] / rates["impressions"] * 100).clip(upper=100).round(1)
    rates["Click Rate %"] = (rates["clicks"] / rates["widget_views"].replace(0, 1) * 100).clip(upper=100).round(2)
    rates["Search Rate %"] = (rates["searches"] / rates["impressions"] * 100).clip(upper=100).round(2)

    col_vr, col_cr, col_sr = st.columns(3)
    for col_container, rate_col, color in [
        (col_vr, "View Rate %", COLORS["primary"]),
        (col_cr, "Click Rate %", COLORS["primary_dark"]),
        (col_sr, "Search Rate %", COLORS["dark_grey"]),
    ]:
        with col_container:
            sorted_rates = rates[[rate_col]].sort_values(rate_col, ascending=True)
            fig_r = go.Figure(go.Bar(
                y=sorted_rates.index,
                x=sorted_rates[rate_col],
                orientation="h",
                marker_color=color,
                text=[f"{v:.1f}%" for v in sorted_rates[rate_col]],
                textposition="auto",
                insidetextanchor="end",
                textfont=dict(size=11),
            ))
            fig_r.update_layout(
                title=dict(text=rate_col, y=0.98),
                height=480,
                margin=dict(l=0, r=10, t=40, b=0),
                xaxis_title="",
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                uniformtext=dict(minsize=9, mode="show"),
            )
            st.plotly_chart(fig_r, use_container_width=True)

    with st.expander("Full UTM breakdown table"):
        st.dataframe(utm_df, use_container_width=True)


# ============================================================
# 6. Top Search Destinations
# ============================================================
st.header("Top Search Destinations")
destinations = top_search_destinations(df)
if not destinations.empty:
    top_dest = destinations.head(20).sort_values("count", ascending=True)
    top_name = destinations.index[0]
    top_count = destinations.iloc[0]["count"]

    fig_dest = go.Figure(go.Bar(
        y=top_dest.index,
        x=top_dest["count"],
        orientation="h",
        marker_color=[
            COLORS["primary"] if name == top_name else COLORS["dark_grey"]
            for name in top_dest.index
        ],
        text=[f'{v:,} ({u} users)' for v, u in zip(top_dest["count"], top_dest["unique_users"])],
        textposition="auto",
        textfont=dict(size=12),
        insidetextanchor="start",
    ))
    fig_dest.update_layout(
        title=f'"{top_name}" leads with {top_count} searches',
        height=max(350, len(top_dest) * 30 + 80),
        margin=dict(l=0, r=40, t=40, b=0),
        uniformtext=dict(minsize=10, mode="show"),
        xaxis_title="Search Count",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
    )
    st.plotly_chart(fig_dest, use_container_width=True)

    # --- Map view of search destinations ---
    dest_names = tuple(destinations.head(20).index.tolist())
    with st.spinner("Geocoding search destinations..."):
        dest_coords = _geocode_locations(dest_names)
    if dest_coords:
        map_rows = []
        for name in dest_names:
            if name in dest_coords:
                lat, lon = dest_coords[name]
                cnt = int(destinations.loc[name, "count"])
                users = int(destinations.loc[name, "unique_users"])
                map_rows.append({"location": name, "lat": lat, "lon": lon,
                                 "searches": cnt, "unique_users": users})
        if map_rows:
            map_df = pd.DataFrame(map_rows)
            fig_map = px.scatter_geo(
                map_df, lat="lat", lon="lon", size="searches",
                hover_name="location",
                hover_data={"searches": True, "unique_users": True, "lat": False, "lon": False},
                color="searches", color_continuous_scale=[[0, "#FFF3B0"], [0.5, "#FFD100"], [1, "#000000"]],
                size_max=40,
                scope="north america",
                title="Where are people searching for travel?",
            )
            fig_map.update_geos(
                showland=True, landcolor="#F0F0F0",
                showlakes=True, lakecolor="white",
                showcountries=True, countrycolor="#CCCCCC",
                showsubunits=True, subunitcolor="#CCCCCC",
            )
            fig_map.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                dragmode=False,
            )
            st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": False})

    with st.expander("Search destinations table"):
        st.dataframe(destinations.head(20), use_container_width=True)


# ============================================================
# 7. Top UTM Terms
# ============================================================
st.header("Top UTM Terms")
utm_terms = df[df["utm_term"].notna()]["utm_term"]
if not utm_terms.empty:
    term_counts = (
        utm_terms.value_counts()
        .head(20)
        .rename_axis("utm_term")
        .reset_index(name="events")
    )
    # Horizontal bar sorted by value — much easier to read than vertical
    fig_terms = go.Figure(go.Bar(
        y=term_counts["utm_term"],
        x=term_counts["events"],
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{v:,}" for v in term_counts["events"]],
        textposition="outside",
    ))
    fig_terms.update_layout(
        title=f'"{term_counts.iloc[0]["utm_term"]}" is the most common UTM term',
        height=max(400, len(term_counts) * 28 + 80),
        margin=dict(l=0, r=60, t=40, b=0),
        xaxis_title="Events",
        yaxis_title="",
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
    )
    st.plotly_chart(fig_terms, use_container_width=True)

    with st.expander("UTM terms table"):
        st.dataframe(term_counts.set_index("utm_term"), use_container_width=True)

    # --- Map view: extract locations from "Lodging Near X" pattern ---
    st.subheader("UTM Term Locations Map")
    st.markdown('Locations extracted from "Lodging Near ..." UTM terms — shows where ad traffic is targeting.')
    all_terms = df[df["utm_term"].notna()]["utm_term"]
    term_locations = (
        all_terms
        .apply(lambda t: re.sub(r"^Lodging Near\s+", "", t) if "Lodging Near" in str(t) else None)
        .dropna()
    )
    if not term_locations.empty:
        loc_counts = term_locations.value_counts().head(30)
        loc_names = tuple(loc_counts.index.tolist())
        with st.spinner("Geocoding UTM term locations..."):
            loc_coords = _geocode_locations(loc_names)
        if loc_coords:
            map_rows = []
            for name in loc_names:
                if name in loc_coords:
                    lat, lon = loc_coords[name]
                    map_rows.append({"location": name, "lat": lat, "lon": lon,
                                     "events": int(loc_counts[name])})
            if map_rows:
                loc_map_df = pd.DataFrame(map_rows)
                fig_loc_map = px.scatter_geo(
                    loc_map_df, lat="lat", lon="lon", size="events",
                    hover_name="location",
                    hover_data={"events": True, "lat": False, "lon": False},
                    color="events", color_continuous_scale=[[0, "#FFF3B0"], [0.5, "#FFD100"], [1, "#000000"]],
                    size_max=35,
                    scope="north america",
                    title="Where is ad traffic targeting?",
                )
                fig_loc_map.update_geos(
                    showland=True, landcolor="#F0F0F0",
                    showlakes=True, lakecolor="white",
                    showcountries=True, countrycolor="#CCCCCC",
                    showsubunits=True, subunitcolor="#CCCCCC",
                )
                fig_loc_map.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0),
                    dragmode=False,
                )
                st.plotly_chart(fig_loc_map, use_container_width=True, config={"scrollZoom": False})
else:
    st.info("No UTM terms found in this date range.")


# ============================================================
# 8. Registered Users
# ============================================================
st.header("Registered Users")
st.markdown("All registered users (with `user_id`) who engaged with the travel widget.")

registered_events = df[df["user_id"].notna()]
if registered_events.empty:
    st.info("No registered users found in this date range.")
else:
    reg_summary = (
        registered_events.groupby("user_id").agg(
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
            total_events=("event", "size"),
            impressions=("event", lambda x: (x == "Loaded a Page").sum()),
            widget_views=("event", lambda x: (x == "Embed Widget Viewed").sum()),
            clicks=("event", lambda x: (x == "Embed Widget Clicked").sum()),
            searches=("event", lambda x: (x == "Travel Searched").sum()),
            sessions=("session_id", "nunique"),
            utm_sources=("utm_source", lambda x: ", ".join(x.dropna().unique())),
        )
        .sort_values("total_events", ascending=False)
    )
    st.metric("Registered Users", f"{len(reg_summary):,}")
    st.dataframe(reg_summary, use_container_width=True)

    csv_buf = io.StringIO()
    reg_summary.reset_index().to_csv(csv_buf, index=False)
    st.download_button(
        label="Download Registered Users CSV",
        data=csv_buf.getvalue(),
        file_name=f"registered_users_{start_date}_{end_date}.csv",
        mime="text/csv",
    )


# ============================================================
# 9. Anonymous User Analysis — activation opportunities
# ============================================================
st.header("Anonymous User Analysis")
st.markdown(
    "Anonymous visitors segmented by engagement level. "
    "**Repeat visitors who engaged with the widget are your best activation targets.**"
)

viewers = engaged_users(df)
if viewers.empty:
    st.info("No engaged anonymous users found.")
else:
    user_ids_set = set(df[df["user_id"].notna()]["anonymous_id"].unique())
    anon_viewers = viewers.loc[~viewers.index.isin(user_ids_set)].copy()
    repeat_engaged = anon_viewers[anon_viewers["sessions"] > 1].copy()
    single_engaged = anon_viewers[anon_viewers["sessions"] == 1].copy()

    # KPI row
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Total Engaged Anonymous", f"{len(anon_viewers):,}")
    col_r2.metric("Repeat Visitors (2+ sessions)", f"{len(repeat_engaged):,}")
    col_r3.metric("Single-Visit Engaged", f"{len(single_engaged):,}")
    repeat_engaged_pct = len(repeat_engaged) / len(anon_viewers) * 100 if len(anon_viewers) > 0 else 0
    col_r4.metric(
        "Repeat Engaged Rate",
        f"{repeat_engaged_pct:.1f}%",
    )

    # Breakdown by max funnel step: where do repeat anonymous users get to?
    if not repeat_engaged.empty and "max_funnel_step" in repeat_engaged.columns:
        st.subheader("Repeat Anonymous Users by Deepest Funnel Step")
        st.markdown("Where do your most engaged anonymous users stop in the funnel?")
        step_order = [
            "Embed Widget Viewed", "Embed Widget Clicked",
            "Travel Searched", "Property Added to Cart", "Checkout Clicked",
        ]
        step_counts = repeat_engaged["max_funnel_step"].value_counts()
        step_counts = step_counts.reindex([s for s in step_order if s in step_counts.index])

        fig_steps = go.Figure(go.Bar(
            x=step_counts.index,
            y=step_counts.values,
            marker_color=[
                COLORS["primary"] if step in ("Travel Searched", "Property Added to Cart", "Checkout Clicked")
                else COLORS["dark_grey"]
                for step in step_counts.index
            ],
            text=[f"{v:,}" for v in step_counts.values],
            textposition="outside",
        ))
        fig_steps.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Deepest Funnel Step Reached",
            yaxis_title="Users",
            showlegend=False,
        )
        st.plotly_chart(fig_steps, use_container_width=True)

    # Session distribution of repeat anonymous
    if not repeat_engaged.empty:
        st.subheader("Session Frequency of Repeat Anonymous Users")
        session_dist = repeat_engaged["sessions"].value_counts().sort_index()
        fig_sess = go.Figure(go.Bar(
            x=session_dist.index.astype(str),
            y=session_dist.values,
            marker_color=COLORS["primary"],
            text=[f"{v:,}" for v in session_dist.values],
            textposition="outside",
        ))
        mode_sess = int(session_dist.idxmax())
        mode_count = int(session_dist.max())
        fig_sess.update_layout(
            title=f"{mode_sess} sessions is the most common return frequency — {mode_count:,} users",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="Number of Sessions",
            yaxis_title="Users",
            showlegend=False,
        )
        st.plotly_chart(fig_sess, use_container_width=True)

    # Where do repeat anonymous users come from?
    if not repeat_engaged.empty and "utm_sources" in repeat_engaged.columns:
        st.subheader("Repeat Anonymous Users by Traffic Source")
        st.markdown(
            "Which sites are driving repeat anonymous visitors? These users may already be "
            "registered on their source site (e.g., BikeReg) — they're not truly unknown, "
            "just not identified on the travel site yet."
        )
        source_counts = (
            repeat_engaged["utm_sources"]
            .str.split(", ")
            .explode()
            .value_counts()
            .head(15)
        )
        source_sorted = source_counts.sort_values(ascending=True)
        fig_src = go.Figure(go.Bar(
            y=source_sorted.index,
            x=source_sorted.values,
            orientation="h",
            marker_color=[
                COLORS["primary"] if name == source_sorted.index[-1] else COLORS["dark_grey"]
                for name in source_sorted.index
            ],
            text=[f"{v:,}" for v in source_sorted.values],
            textposition="outside",
        ))
        fig_src.update_layout(
            title=f"{source_sorted.index[-1]} leads with {source_sorted.values[-1]:,} repeat anonymous users",
            height=max(300, len(source_sorted) * 35 + 80),
            margin=dict(l=0, r=60, t=40, b=0),
            xaxis_title="Repeat Anonymous Users",
            yaxis_title="",
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
        )
        st.plotly_chart(fig_src, use_container_width=True)

    # Data table
    st.subheader("Repeat Anonymous Users (Top 50)")
    if not repeat_engaged.empty:
        display_cols = [
            c for c in ["first_seen", "last_seen", "total_events", "widget_views",
                        "widget_clicks", "searches", "sessions", "utm_sources",
                        "referrers", "max_funnel_step"]
            if c in repeat_engaged.columns
        ]
        st.dataframe(repeat_engaged[display_cols].head(50), use_container_width=True)

    # Exports
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_buf = io.StringIO()
        repeat_engaged.reset_index().to_csv(csv_buf, index=False)
        st.download_button(
            label="Download Repeat Anonymous Users CSV",
            data=csv_buf.getvalue(),
            file_name=f"repeat_anonymous_{start_date}_{end_date}.csv",
            mime="text/csv",
            type="primary",
        )
    with col_dl2:
        csv_buf2 = io.StringIO()
        anon_viewers.reset_index().to_csv(csv_buf2, index=False)
        st.download_button(
            label="Download All Anonymous Users CSV",
            data=csv_buf2.getvalue(),
            file_name=f"anonymous_users_{start_date}_{end_date}.csv",
            mime="text/csv",
        )


# ============================================================
# 10. Activation Strategy Recommendations
# ============================================================
st.header("Activation Strategy")
st.markdown(
    "Based on the data above, here are recommended strategies to convert "
    "anonymous visitors into registered users and drive deeper engagement."
)

top_dest_name = destinations.index[0] if not destinations.empty else "top destinations"

_STRATEGY_CARD_CSS = """
<div style="
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-top: 4px solid {color};
    border-radius: 10px;
    padding: 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    height: 100%;
">
<h4 style="margin-top:0; color:#000000;">{title}</h4>
<p style="color:#555555; font-size:0.95rem;">{description}</p>
<ul style="color:#333333; font-size:0.9rem; padding-left:20px;">
{items}
</ul>
</div>
"""

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown(_STRATEGY_CARD_CSS.format(
        color="#FFD100",
        title="Capture Repeat Anonymous Users",
        description=f"{repeat_anon:,} users returned multiple times without registering. Many are already known on sister sites (BikeReg, SkiReg). Give them a reason to share their email.",
        items="<li><b>2nd-visit registration nudge:</b> Detect returning visitors and prompt: &quot;Create an account for exclusive travel deals.&quot;</li><li><b>Email capture via content:</b> Offer trip planning checklists, packing guides, or destination newsletters in exchange for email sign-up.</li>",
    ), unsafe_allow_html=True)

with col_s2:
    st.markdown(_STRATEGY_CARD_CSS.format(
        color="#E6BC00",
        title="Destination-Targeted Lead Magnets",
        description=f"Searches concentrate on a few destinations (e.g., {top_dest_name}). Use proven demand to capture emails.",
        items="<li><b>Destination guides:</b> &quot;Download our insider guide to skiing in Banff&quot; — gated behind email. Create guides for top-searched destinations.</li><li>Create dedicated landing pages for top-searched destinations</li><li>Run email campaigns with deals for these locations</li>",
    ), unsafe_allow_html=True)

col_s3, col_s4 = st.columns(2)

with col_s3:
    st.markdown(_STRATEGY_CARD_CSS.format(
        color="#000000",
        title="Optimize the Widget CTR",
        description="Widget CTR is 0.05% vs. 0.47% travel industry average. Even small improvements at 60K+ views would have outsized impact.",
        items="<li>A/B test widget headline, CTA copy, and page placement</li><li>Add social proof (e.g., &quot;12 people searched Aspen today&quot;)</li>",
    ), unsafe_allow_html=True)

with col_s4:
    st.markdown(_STRATEGY_CARD_CSS.format(
        color="#333333",
        title="Cross-Site Identity Resolution",
        description="Most repeat anonymous users come from BikeReg and SkiReg where they may already be registered. Stitch identities to unlock email-addressable audiences.",
        items="<li>Investigate RudderStack Profiles to merge anonymous_id with sister site registrations</li><li>Cross-reference AREG user databases (BikeReg, SkiReg, RunReg, TriReg) with travel widget impression data</li>",
    ), unsafe_allow_html=True)
