"""Generate a self-contained static HTML page replicating the Streamlit dashboard.

Usage:
    python generate_html.py

Outputs:
    docs/index.html
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from travel_events.analysis import (
    daily_event_counts,
    engaged_users,
    event_summary,
    funnel,
    top_search_destinations,
    utm_breakdown,
)

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#FFD100",
    "primary_dark": "#E6BC00",
    "primary_light": "#FFF3B0",
    "black": "#000000",
    "dark_grey": "#333333",
    "mid_grey": "#666666",
    "grey": "#999999",
    "light_grey": "#CCCCCC",
    "bg_grey": "#F7F7F7",
    "accent": "#FFD100",
    "secondary": "#333333",
}

CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
GEOCACHE_PATH = _PROJECT_ROOT / "data" / "geocache.json"
OUTPUT_PATH = _PROJECT_ROOT / "docs" / "index.html"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_parquet() -> pd.DataFrame:
    """Load all parquet files from data/cache/."""
    frames = []
    for path in sorted(CACHE_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise RuntimeError("No parquet files found in data/cache/")
    # Drop all-NA columns before concat to avoid FutureWarning
    frames = [f.dropna(axis=1, how="all") for f in frames]
    return pd.concat(frames, ignore_index=True)


def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same data transforms as the Streamlit dashboard."""
    # Guard against "None" strings from Parquet cache coercion
    if "user_id" in df.columns:
        df.loc[df["user_id"].isin(["None", "none", ""]), "user_id"] = None
        # Remove internal users (Outside Inc. and Tenex employees)
        internal_mask = df["user_id"].str.contains(r"@outsideinc\.com|@tenex\.co", na=False)
        df.loc[internal_mask, "user_id"] = None

    # Fill missing utm_source from referring_domain
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
        if "utm_campaign" in df.columns:
            df["utm_campaign"] = df["utm_campaign"].fillna("(none)")
    return df


def load_geocache() -> dict:
    """Load geocache from disk."""
    if GEOCACHE_PATH.exists():
        try:
            return json.loads(GEOCACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def fig_to_html(fig, config=None) -> str:
    """Convert a Plotly figure to an HTML div (no full page, no plotly.js)."""
    config = config or {}
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=config,
    )


def kpi_card(label: str, value: str) -> str:
    return f"""<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
</div>"""


def strategy_card(color: str, title: str, description: str, items: str) -> str:
    return f"""<div style="
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
</div>"""


def df_to_html_table(df: pd.DataFrame, max_rows: int = None) -> str:
    """Convert a DataFrame to a styled HTML table."""
    display_df = df.head(max_rows) if max_rows else df
    html = display_df.to_html(classes="data-table", border=0, na_rep="")
    return f'<div class="table-wrapper">{html}</div>'


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate():
    print("Loading parquet data...")
    df = load_all_parquet()
    df = apply_transforms(df)

    if df.empty:
        print("ERROR: No data loaded.")
        return

    geocache = load_geocache()

    # Date range from cached parquet files (not event timestamps, which can
    # include late-arriving events from earlier dates)
    parquet_dates = sorted(f.stem for f in CACHE_DIR.glob("*.parquet"))
    min_date = parquet_dates[0] if parquet_dates else str(df["timestamp"].min().date())
    max_date = parquet_dates[-1] if parquet_dates else str(df["timestamp"].max().date())

    # -----------------------------------------------------------------------
    # 1. Audience Overview
    # -----------------------------------------------------------------------
    total_unique = df["anonymous_id"].nunique()
    registered_count = df["user_id"].dropna().nunique()
    ids_with_uid = set(df[df["user_id"].notna()]["anonymous_id"].unique())
    anon_df = df[~df["anonymous_id"].isin(ids_with_uid) & df["user_id"].isna()]
    anon_count = anon_df["anonymous_id"].nunique()
    anon_sessions = anon_df.groupby("anonymous_id")["session_id"].nunique()
    single_visit_anon = int((anon_sessions == 1).sum())
    repeat_anon = int((anon_sessions > 1).sum())
    repeat_pct = repeat_anon / anon_count * 100 if anon_count > 0 else 0

    kpi_row1 = "".join([
        kpi_card("Total Events", f"{len(df):,}"),
        kpi_card("Unique Event Types", f"{df['event'].nunique()}"),
        kpi_card("Unique Users", f"{total_unique:,}"),
        kpi_card("Registered Users", f"{registered_count:,}"),
    ])
    kpi_row2 = "".join([
        kpi_card("Anonymous Users", f"{anon_count:,}"),
        kpi_card("Single-Visit Anon.", f"{single_visit_anon:,}"),
        kpi_card("Repeat Anonymous", f"{repeat_anon:,}"),
        kpi_card("Repeat Rate", f"{repeat_pct:.1f}%"),
    ])

    # -----------------------------------------------------------------------
    # User Composition pie chart
    # -----------------------------------------------------------------------
    total_pie = registered_count + repeat_anon + single_visit_anon
    reg_pct = registered_count / total_pie * 100 if total_pie > 0 else 0
    single_pct = single_visit_anon / total_pie * 100 if total_pie > 0 else 0
    repeat_pct_pie = repeat_anon / total_pie * 100 if total_pie > 0 else 0

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
            "",
        ],
        textposition="outside",
        textfont=dict(size=14, color="#000000"),
        pull=[0, 0.04, 0.15],
        sort=False,
        direction="clockwise",
    ))
    fig_comp.add_annotation(
        text=f"<b>Registered: {registered_count:,} ({reg_pct:.2f}%)</b>",
        x=0.52, y=1.0,
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=COLORS["black"],
        ax=80, ay=30,
        font=dict(size=13, color=COLORS["black"]),
        bgcolor=COLORS["primary"], bordercolor=COLORS["black"],
        borderwidth=1, borderpad=6,
    )
    fig_comp.update_layout(height=400, margin=dict(l=40, r=40, t=30, b=30), showlegend=False)
    comp_html = fig_to_html(fig_comp)

    # -----------------------------------------------------------------------
    # 2. Event Summary
    # -----------------------------------------------------------------------
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
        title="Page loads dominate -- engagement events are <1% of traffic",
        height=max(300, len(summary_sorted) * 40 + 80),
        margin=dict(l=0, r=80, t=40, b=0),
        xaxis_title="Event Count", yaxis_title="", showlegend=False,
    )
    events_html = fig_to_html(fig_events)
    events_table_html = df_to_html_table(summary)

    # -----------------------------------------------------------------------
    # 3. Daily Event Trend
    # -----------------------------------------------------------------------
    daily = daily_event_counts(df)

    high_vol_cols = [c for c in ["Loaded a Page", "Embed Widget Viewed"] if c in daily.columns]
    low_vol_cols = [c for c in daily.columns if c not in high_vol_cols and daily[c].sum() > 0]

    high_vol_html = ""
    if high_vol_cols:
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
        high_vol_html = fig_to_html(fig_high)

    # Heatmap
    # Heatmap removed — without tied-in external events, the daily breakdown
    # creates noise rather than actionable insight for stakeholders.

    # -----------------------------------------------------------------------
    # 4. Conversion Funnel
    # -----------------------------------------------------------------------
    funnel_df = funnel(df)
    funnel_html = ""
    funnel_table_html = ""
    if not funnel_df.empty:
        fig_funnel_anon = go.Figure(go.Funnel(
            y=funnel_df["step"], x=funnel_df["anonymous_users"],
            textinfo="value+percent initial",
            marker_color=COLORS["dark_grey"],
            textfont=dict(size=12),
        ))
        fig_funnel_anon.update_layout(
            title="Anonymous Users", height=420,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        fig_funnel_reg = go.Figure(go.Funnel(
            y=funnel_df["step"], x=funnel_df["registered_users"],
            textinfo="value+percent initial",
            marker_color=COLORS["primary"],
            textfont=dict(size=12),
        ))
        fig_funnel_reg.update_layout(
            title="Registered Users", height=420,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        funnel_html = f"""<div class="two-col">
    <div>{fig_to_html(fig_funnel_anon)}</div>
    <div>{fig_to_html(fig_funnel_reg)}</div>
</div>"""
        funnel_table_html = df_to_html_table(funnel_df)

    # -----------------------------------------------------------------------
    # 5. UTM Performance
    # -----------------------------------------------------------------------
    utm_df = utm_breakdown(df)
    utm_volume_html = ""
    utm_rates_html = ""
    utm_table_html = ""

    if not utm_df.empty:
        utm_flat = utm_df.reset_index()
        by_source = (
            utm_flat.groupby("utm_source")[["impressions", "widget_views", "clicks", "searches"]]
            .sum()
            .sort_values("impressions", ascending=False)
            .head(15)
        )
        by_source_asc = by_source.sort_values("impressions", ascending=True)
        fixed_y_order = by_source_asc.index.tolist()

        # Volume chart: page loads & widget views
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

        # Widget clicks chart
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

        utm_volume_html = f"""<div class="two-col">
    <div>{fig_to_html(fig_vol)}</div>
    <div>{fig_to_html(fig_eng)}</div>
</div>"""

        # Conversion rates
        rates = by_source.copy()
        rates["View Rate %"] = (rates["widget_views"] / rates["impressions"] * 100).clip(upper=100).round(1)
        rates["Click Rate %"] = (rates["clicks"] / rates["widget_views"].replace(0, 1) * 100).clip(upper=100).round(2)
        rates["Search Rate %"] = (rates["searches"] / rates["impressions"] * 100).clip(upper=100).round(2)

        rate_figs = []
        for rate_col, color in [
            ("View Rate %", COLORS["primary"]),
            ("Click Rate %", COLORS["primary_dark"]),
            ("Search Rate %", COLORS["dark_grey"]),
        ]:
            sorted_rates = rates[[rate_col]].sort_values(rate_col, ascending=True)
            fig_r = go.Figure(go.Bar(
                y=sorted_rates.index, x=sorted_rates[rate_col],
                orientation="h", marker_color=color,
                text=[f"{v:.1f}%" for v in sorted_rates[rate_col]],
                textposition="auto", insidetextanchor="end",
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
            rate_figs.append(fig_to_html(fig_r))

        utm_rates_html = f"""<div class="three-col">
    <div>{rate_figs[0]}</div>
    <div>{rate_figs[1]}</div>
    <div>{rate_figs[2]}</div>
</div>"""

        utm_table_html = df_to_html_table(utm_df)

    # -----------------------------------------------------------------------
    # 6. Top Search Destinations
    # -----------------------------------------------------------------------
    destinations = top_search_destinations(df)
    dest_bar_html = ""
    dest_map_html = ""
    dest_table_html = ""
    top_dest_name = ""

    if not destinations.empty:
        top_dest = destinations.head(20).sort_values("count", ascending=True)
        top_dest_name = destinations.index[0]
        top_count = destinations.iloc[0]["count"]

        fig_dest = go.Figure(go.Bar(
            y=top_dest.index, x=top_dest["count"],
            orientation="h",
            marker_color=[
                COLORS["primary"] if name == top_dest_name else COLORS["dark_grey"]
                for name in top_dest.index
            ],
            text=[f'{v:,} ({u} users)' for v, u in zip(top_dest["count"], top_dest["unique_users"])],
            textposition="auto", textfont=dict(size=12),
            insidetextanchor="start",
        ))
        fig_dest.update_layout(
            title=f'"{top_dest_name}" leads with {top_count} searches',
            height=max(350, len(top_dest) * 30 + 80),
            margin=dict(l=0, r=40, t=40, b=0),
            uniformtext=dict(minsize=10, mode="show"),
            xaxis_title="Search Count", yaxis_title="",
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
        )
        dest_bar_html = fig_to_html(fig_dest)

        # Map
        dest_names = destinations.head(20).index.tolist()
        map_rows = []
        for name in dest_names:
            if name in geocache and geocache[name] is not None:
                lat, lon = geocache[name]
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
                color="searches",
                color_continuous_scale=[[0, "#FFF3B0"], [0.5, "#FFD100"], [1, "#000000"]],
                size_max=40, scope="north america",
                title="Where are people searching for travel?",
            )
            fig_map.update_geos(
                showland=True, landcolor="#F0F0F0",
                showlakes=True, lakecolor="white",
                showcountries=True, countrycolor="#CCCCCC",
                showsubunits=True, subunitcolor="#CCCCCC",
            )
            fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0), dragmode=False)
            dest_map_html = fig_to_html(fig_map, config={"scrollZoom": False})

        dest_table_html = df_to_html_table(destinations.head(20))

    # -----------------------------------------------------------------------
    # 7. Top UTM Terms
    # -----------------------------------------------------------------------
    utm_terms_series = df[df["utm_term"].notna()]["utm_term"] if "utm_term" in df.columns else pd.Series(dtype=str)
    utm_terms_bar_html = ""
    utm_terms_table_html = ""
    utm_terms_map_html = ""

    if not utm_terms_series.empty:
        term_counts = (
            utm_terms_series.value_counts()
            .head(20)
            .rename_axis("utm_term")
            .reset_index(name="events")
        )

        fig_terms = go.Figure(go.Bar(
            y=term_counts["utm_term"], x=term_counts["events"],
            orientation="h", marker_color=COLORS["primary"],
            text=[f"{v:,}" for v in term_counts["events"]],
            textposition="outside",
        ))
        fig_terms.update_layout(
            title=f'"{term_counts.iloc[0]["utm_term"]}" is the most common UTM term',
            height=max(400, len(term_counts) * 28 + 80),
            margin=dict(l=0, r=60, t=40, b=0),
            xaxis_title="Events", yaxis_title="",
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
        )
        utm_terms_bar_html = fig_to_html(fig_terms)
        utm_terms_table_html = df_to_html_table(term_counts.set_index("utm_term"))

        # UTM term locations map
        all_terms = df[df["utm_term"].notna()]["utm_term"]
        term_locations = (
            all_terms
            .apply(lambda t: re.sub(r"^Lodging Near\s+", "", t) if "Lodging Near" in str(t) else None)
            .dropna()
        )
        if not term_locations.empty:
            loc_counts = term_locations.value_counts().head(30)
            loc_names = loc_counts.index.tolist()
            map_rows = []
            for name in loc_names:
                if name in geocache and geocache[name] is not None:
                    lat, lon = geocache[name]
                    map_rows.append({"location": name, "lat": lat, "lon": lon,
                                     "events": int(loc_counts[name])})
            if map_rows:
                loc_map_df = pd.DataFrame(map_rows)
                fig_loc_map = px.scatter_geo(
                    loc_map_df, lat="lat", lon="lon", size="events",
                    hover_name="location",
                    hover_data={"events": True, "lat": False, "lon": False},
                    color="events",
                    color_continuous_scale=[[0, "#FFF3B0"], [0.5, "#FFD100"], [1, "#000000"]],
                    size_max=35, scope="north america",
                    title="Where is ad traffic targeting?",
                )
                fig_loc_map.update_geos(
                    showland=True, landcolor="#F0F0F0",
                    showlakes=True, lakecolor="white",
                    showcountries=True, countrycolor="#CCCCCC",
                    showsubunits=True, subunitcolor="#CCCCCC",
                )
                fig_loc_map.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0), dragmode=False)
                utm_terms_map_html = fig_to_html(fig_loc_map, config={"scrollZoom": False})

    # -----------------------------------------------------------------------
    # 8. Registered Users table
    # -----------------------------------------------------------------------
    registered_events = df[df["user_id"].notna()]
    reg_table_html = ""
    reg_user_count = 0
    if not registered_events.empty:
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
        reg_user_count = len(reg_summary)
        reg_table_html = df_to_html_table(reg_summary)

    # -----------------------------------------------------------------------
    # 9. Anonymous User Analysis
    # -----------------------------------------------------------------------
    viewers = engaged_users(df)
    anon_analysis_html = ""
    if not viewers.empty:
        user_ids_set = set(df[df["user_id"].notna()]["anonymous_id"].unique())
        anon_viewers = viewers.loc[~viewers.index.isin(user_ids_set)].copy()
        repeat_engaged = anon_viewers[anon_viewers["sessions"] > 1].copy()
        single_engaged = anon_viewers[anon_viewers["sessions"] == 1].copy()
        repeat_engaged_pct = len(repeat_engaged) / len(anon_viewers) * 100 if len(anon_viewers) > 0 else 0

        # KPI row
        anon_kpi_row = "".join([
            kpi_card("Total Engaged Anonymous", f"{len(anon_viewers):,}"),
            kpi_card("Repeat Visitors (2+ sessions)", f"{len(repeat_engaged):,}"),
            kpi_card("Single-Visit Engaged", f"{len(single_engaged):,}"),
            kpi_card("Repeat Engaged Rate", f"{repeat_engaged_pct:.1f}%"),
        ])

        # Funnel steps chart
        funnel_steps_html = ""
        if not repeat_engaged.empty and "max_funnel_step" in repeat_engaged.columns:
            step_order = [
                "Embed Widget Viewed", "Embed Widget Clicked",
                "Travel Searched", "Property Added to Cart", "Checkout Clicked",
            ]
            step_counts = repeat_engaged["max_funnel_step"].value_counts()
            step_counts = step_counts.reindex([s for s in step_order if s in step_counts.index])

            fig_steps = go.Figure(go.Bar(
                x=step_counts.index, y=step_counts.values,
                marker_color=[
                    COLORS["primary"] if step in ("Travel Searched", "Property Added to Cart", "Checkout Clicked")
                    else COLORS["dark_grey"]
                    for step in step_counts.index
                ],
                text=[f"{v:,}" for v in step_counts.values],
                textposition="outside",
            ))
            fig_steps.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Deepest Funnel Step Reached", yaxis_title="Users",
                showlegend=False,
            )
            funnel_steps_html = f"""
<h3>Repeat Anonymous Users by Deepest Funnel Step</h3>
<p>Where do your most engaged anonymous users stop in the funnel?</p>
{fig_to_html(fig_steps)}
"""

        # Session distribution
        session_dist_html = ""
        if not repeat_engaged.empty:
            session_dist = repeat_engaged["sessions"].value_counts().sort_index()
            fig_sess = go.Figure(go.Bar(
                x=session_dist.index.astype(str), y=session_dist.values,
                marker_color=COLORS["primary"],
                text=[f"{v:,}" for v in session_dist.values],
                textposition="outside",
            ))
            mode_sess = int(session_dist.idxmax())
            mode_count = int(session_dist.max())
            fig_sess.update_layout(
                title=f"{mode_sess} sessions is the most common return frequency -- {mode_count:,} users",
                height=300, margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Number of Sessions", yaxis_title="Users",
                showlegend=False,
            )
            session_dist_html = f"""
<h3>Session Frequency of Repeat Anonymous Users</h3>
{fig_to_html(fig_sess)}
"""

        # Traffic source chart
        traffic_src_html = ""
        if not repeat_engaged.empty and "utm_sources" in repeat_engaged.columns:
            source_counts = (
                repeat_engaged["utm_sources"]
                .str.split(", ")
                .explode()
                .value_counts()
                .head(15)
            )
            source_sorted = source_counts.sort_values(ascending=True)
            fig_src = go.Figure(go.Bar(
                y=source_sorted.index, x=source_sorted.values,
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
                xaxis_title="Repeat Anonymous Users", yaxis_title="",
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
            )
            traffic_src_html = f"""
<h3>Repeat Anonymous Users by Traffic Source</h3>
<p>Which sites are driving repeat anonymous visitors? These users may already be
registered on their source site (e.g., BikeReg) -- they're not truly unknown,
just not identified on the travel site yet.</p>
{fig_to_html(fig_src)}
"""

        # Repeat anonymous table
        repeat_table_html = ""
        if not repeat_engaged.empty:
            display_cols = [
                c for c in ["first_seen", "last_seen", "total_events", "widget_views",
                            "widget_clicks", "searches", "sessions", "utm_sources",
                            "referrers", "max_funnel_step"]
                if c in repeat_engaged.columns
            ]
            repeat_table_html = f"""
<h3>Repeat Anonymous Users (Top 50)</h3>
{df_to_html_table(repeat_engaged[display_cols], max_rows=50)}
"""

        anon_analysis_html = f"""
<div class="kpi-grid">{anon_kpi_row}</div>
{funnel_steps_html}
{session_dist_html}
{traffic_src_html}
{repeat_table_html}
"""

    # -----------------------------------------------------------------------
    # 10. Activation Strategy
    # -----------------------------------------------------------------------
    top_dest_label = top_dest_name if top_dest_name else "top destinations"

    strategy_cards = f"""<div class="two-col" style="margin-bottom:20px;">
    <div>{strategy_card(
        "#FFD100",
        "Capture Repeat Anonymous Users",
        f"{repeat_anon:,} users returned multiple times without registering. Many are already known on sister sites (BikeReg, SkiReg). Give them a reason to share their email.",
        '<li><b>2nd-visit registration nudge:</b> Detect returning visitors and prompt: "Create an account for exclusive travel deals."</li><li><b>Email capture via content:</b> Offer trip planning checklists, packing guides, or destination newsletters in exchange for email sign-up.</li>',
    )}</div>
    <div>{strategy_card(
        "#E6BC00",
        "Destination-Targeted Lead Magnets",
        f'Searches concentrate on a few destinations (e.g., {top_dest_label}). Use proven demand to capture emails.',
        '<li><b>Destination guides:</b> "Download our insider guide to skiing in Banff" -- gated behind email. Create guides for top-searched destinations.</li><li>Create dedicated landing pages for top-searched destinations</li><li>Run email campaigns with deals for these locations</li>',
    )}</div>
</div>
<div class="two-col">
    <div>{strategy_card(
        "#000000",
        "Optimize the Widget CTR",
        "Widget CTR is 0.05% vs. 0.47% travel industry average. Even small improvements at 60K+ views would have outsized impact.",
        '<li>A/B test widget headline, CTA copy, and page placement</li><li>Add social proof (e.g., "12 people searched Aspen today")</li>',
    )}</div>
    <div>{strategy_card(
        "#333333",
        "Cross-Site Identity Resolution",
        "Most repeat anonymous users come from BikeReg and SkiReg where they may already be registered. Stitch identities to unlock email-addressable audiences.",
        '<li>Investigate RudderStack Profiles to merge anonymous_id with sister site registrations</li><li>Cross-reference AREG user databases (BikeReg, SkiReg, RunReg, TriReg) with travel widget impression data</li>',
    )}</div>
</div>"""

    # -----------------------------------------------------------------------
    # Assemble full HTML
    # -----------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Outside Travel Insights</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: {COLORS["bg_grey"]};
        color: {COLORS["dark_grey"]};
        line-height: 1.6;
        padding: 0;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px 24px;
    }}
    h1 {{
        color: {COLORS["black"]};
        font-size: 2.2rem;
        margin-bottom: 8px;
        border-bottom: 4px solid {COLORS["primary"]};
        padding-bottom: 12px;
    }}
    h2 {{
        color: {COLORS["black"]};
        font-size: 1.5rem;
        margin-top: 48px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid {COLORS["primary"]};
    }}
    h3 {{
        color: {COLORS["dark_grey"]};
        font-size: 1.2rem;
        margin-top: 28px;
        margin-bottom: 8px;
    }}
    h4 {{
        color: {COLORS["dark_grey"]};
        font-size: 1.05rem;
        margin-top: 20px;
        margin-bottom: 8px;
    }}
    p {{ margin-bottom: 12px; }}
    ul {{ margin-left: 20px; margin-bottom: 12px; }}
    li {{ margin-bottom: 6px; }}
    .caption {{
        font-size: 0.85rem;
        color: {COLORS["grey"]};
        margin-top: 4px;
    }}
    .subtitle {{
        font-size: 0.95rem;
        color: {COLORS["mid_grey"]};
        margin-bottom: 24px;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 16px;
    }}
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-top: 3px solid {COLORS["primary"]};
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .kpi-label {{
        font-size: 0.85rem;
        color: {COLORS["mid_grey"]};
        margin-bottom: 4px;
    }}
    .kpi-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLORS["black"]};
    }}
    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 16px;
    }}
    .three-col {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 20px;
        margin-bottom: 16px;
    }}
    .chart-container {{
        background: #FFFFFF;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }}
    .table-wrapper {{
        overflow-x: auto;
        margin-bottom: 20px;
    }}
    .data-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        background: #FFFFFF;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }}
    .data-table th {{
        background: {COLORS["dark_grey"]};
        color: #FFFFFF;
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
        white-space: nowrap;
    }}
    .data-table td {{
        padding: 8px 12px;
        border-bottom: 1px solid #F0F0F0;
        white-space: nowrap;
    }}
    .data-table tr:hover td {{
        background: {COLORS["primary_light"]};
    }}
    .collapsible-header {{
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 10px 16px;
        cursor: pointer;
        font-weight: 600;
        color: {COLORS["dark_grey"]};
        margin-bottom: 8px;
        user-select: none;
    }}
    .collapsible-header:hover {{
        background: {COLORS["primary_light"]};
    }}
    .collapsible-header::before {{
        content: "\\25B6 ";
        font-size: 0.8em;
        display: inline-block;
        transition: transform 0.2s;
    }}
    .collapsible-header.open::before {{
        transform: rotate(90deg);
    }}
    .collapsible-content {{
        display: none;
        padding: 0 4px;
    }}
    .collapsible-content.open {{
        display: block;
    }}
    /* Responsive */
    @media (max-width: 900px) {{
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
        .two-col {{ grid-template-columns: 1fr; }}
        .three-col {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 600px) {{
        .kpi-grid {{ grid-template-columns: 1fr; }}
        .container {{ padding: 12px; }}
        h1 {{ font-size: 1.6rem; }}
    }}
    /* Plotly chart responsiveness */
    .js-plotly-plot, .plotly {{
        width: 100% !important;
    }}
</style>
</head>
<body>
<div class="container">

<h1>Outside Travel Insights</h1>
<p class="subtitle">Showing data from <strong>{min_date}</strong> to <strong>{max_date}</strong> ({len(df):,} events)</p>

<!-- ============================================================ -->
<!-- 1. Audience Overview -->
<!-- ============================================================ -->
<h2>Audience Overview</h2>
<div class="kpi-grid">{kpi_row1}</div>
<div class="kpi-grid">{kpi_row2}</div>

<h3>User Composition</h3>
{"<p><strong>" + f"{repeat_anon:,} repeat anonymous users</strong> visited multiple times without registering -- these represent your best activation opportunity.</p>" if repeat_anon > 0 else ""}
<div class="chart-container">{comp_html}</div>

<!-- ============================================================ -->
<!-- 2. Event Summary -->
<!-- ============================================================ -->
<h2>Event Summary</h2>
<div class="chart-container">{events_html}</div>
<div class="collapsible-header" onclick="toggleCollapsible(this)">Event breakdown table</div>
<div class="collapsible-content">{events_table_html}</div>

<!-- ============================================================ -->
<!-- 3. Daily Event Trend -->
<!-- ============================================================ -->
<h2>Daily Event Trend</h2>
{"<h3>Page Loads vs Widget Views</h3><div class='chart-container'>" + high_vol_html + "</div>" if high_vol_html else ""}

<!-- ============================================================ -->
<!-- 4. Conversion Funnel -->
<!-- ============================================================ -->
<h2>Conversion Funnel</h2>
{"<p>Separated by user type so registered users are visible on their own scale.</p>" + funnel_html if funnel_html else "<p>No funnel data available.</p>"}
{"<div class='collapsible-header' onclick='toggleCollapsible(this)'>Funnel data table</div><div class='collapsible-content'>" + funnel_table_html + "</div>" if funnel_table_html else ""}

<!-- ============================================================ -->
<!-- 5. UTM Performance -->
<!-- ============================================================ -->
<h2>UTM Performance</h2>
{"<h3>Traffic Volume by UTM Source</h3><p>Page loads &amp; widget views (left) vs. widget clicks (right) -- separate scales so low-volume clicks are visible.</p>" + utm_volume_html if utm_volume_html else ""}

{"<h3>Conversion Rates by UTM Source (Ranked)</h3><p><strong>View Rate</strong> = widget views / page loads (did users scroll to the widget?). <strong>Click Rate</strong> = clicks / widget views (of those who saw it, who clicked?). <strong>Search Rate</strong> = searches / page loads (search is a separate action from the widget).</p>" + utm_rates_html if utm_rates_html else ""}

{"<div class='collapsible-header' onclick='toggleCollapsible(this)'>Full UTM breakdown table</div><div class='collapsible-content'>" + utm_table_html + "</div>" if utm_table_html else ""}

<!-- ============================================================ -->
<!-- 6. Top Search Destinations -->
<!-- ============================================================ -->
<h2>Top Search Destinations</h2>
{"<div class='chart-container'>" + dest_bar_html + "</div>" if dest_bar_html else "<p>No search destination data.</p>"}
{"<div class='chart-container'>" + dest_map_html + "</div>" if dest_map_html else ""}
{"<div class='collapsible-header' onclick='toggleCollapsible(this)'>Search destinations table</div><div class='collapsible-content'>" + dest_table_html + "</div>" if dest_table_html else ""}

<!-- ============================================================ -->
<!-- 7. Top UTM Terms -->
<!-- ============================================================ -->
<h2>Top UTM Terms</h2>
{"<div class='chart-container'>" + utm_terms_bar_html + "</div>" if utm_terms_bar_html else "<p>No UTM terms found in this date range.</p>"}
{"<div class='collapsible-header' onclick='toggleCollapsible(this)'>UTM terms table</div><div class='collapsible-content'>" + utm_terms_table_html + "</div>" if utm_terms_table_html else ""}

{"<h3>UTM Term Locations Map</h3><p>Locations extracted from &quot;Lodging Near ...&quot; UTM terms -- shows where ad traffic is targeting.</p><div class='chart-container'>" + utm_terms_map_html + "</div>" if utm_terms_map_html else ""}

<!-- ============================================================ -->
<!-- 8. Registered Users -->
<!-- ============================================================ -->
<h2>Registered Users</h2>
<p>All registered users (with <code>user_id</code>) who engaged with the travel widget.</p>
{"<div class='kpi-grid'>" + kpi_card("Registered Users", f"{reg_user_count:,}") + "</div>" + reg_table_html if reg_table_html else "<p>No registered users found in this date range.</p>"}

<!-- ============================================================ -->
<!-- 9. Anonymous User Analysis -->
<!-- ============================================================ -->
<h2>Anonymous User Analysis</h2>
<p>Anonymous visitors segmented by engagement level.
<strong>Repeat visitors who engaged with the widget are your best activation targets.</strong></p>
{anon_analysis_html if anon_analysis_html else "<p>No engaged anonymous users found.</p>"}

<!-- ============================================================ -->
<!-- 10. Activation Strategy -->
<!-- ============================================================ -->
<h2>Activation Strategy</h2>
<p>Based on the data above, here are recommended strategies to convert
anonymous visitors into registered users and drive deeper engagement.</p>
{strategy_cards}

</div><!-- .container -->

<script>
function toggleCollapsible(header) {{
    header.classList.toggle('open');
    var content = header.nextElementSibling;
    content.classList.toggle('open');
}}
// Make Plotly charts responsive
window.addEventListener('resize', function() {{
    var plots = document.querySelectorAll('.js-plotly-plot');
    plots.forEach(function(plot) {{
        if (plot && plot.data) {{
            Plotly.Plots.resize(plot);
        }}
    }});
}});
</script>
</body>
</html>"""

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Generated {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    generate()
