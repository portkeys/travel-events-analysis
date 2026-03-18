"""Exploratory analysis helpers for travel widget events."""

import pandas as pd


def event_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count events by type, with anonymous vs registered user breakdown."""
    return (
        df.groupby("event")
        .agg(
            count=("event", "size"),
            unique_users=("anonymous_id", "nunique"),
            registered_users=("user_id", "nunique"),
            anonymous_users=("user_id", lambda x: df.loc[x.index, "anonymous_id"][x.isna()].nunique()),
        )
        .sort_values("count", ascending=False)
    )


def daily_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Daily event counts by type."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    return df.groupby(["date", "event"]).size().unstack(fill_value=0)


def funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Build a funnel with anonymous vs registered user breakdown."""
    funnel_steps = [
        "Loaded a Page",
        "Embed Widget Viewed",
        "Embed Widget Clicked",
        "Travel Searched",
        "Property Added to Cart",
        "Checkout Clicked",
    ]
    rows = []
    for step in funnel_steps:
        step_df = df[df["event"] == step]
        has_uid = step_df[step_df["user_id"].notna()]
        no_uid = step_df[step_df["user_id"].isna()]
        rows.append(
            {
                "step": step,
                "events": len(step_df),
                "unique_users": step_df["anonymous_id"].nunique(),
                "registered_users": has_uid["user_id"].nunique(),
                "anonymous_users": no_uid["anonymous_id"].nunique(),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty and result["unique_users"].iloc[0] > 0:
        result["conversion_rate"] = (
            result["unique_users"] / result["unique_users"].iloc[0] * 100
        ).round(2)
    return result


def utm_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Break down events by UTM source and campaign.

    Shows the four key engagement stages and conversion rates between them:
    - Impressions (Loaded a Page)
    - Widget Views (Embed Widget Viewed) — user scrolled to the widget
    - Clicks (Embed Widget Clicked) — user interacted with the widget
    - Searches (Travel Searched) — user showed booking intent
    """
    result = (
        df.groupby(["utm_source", "utm_campaign"])
        .agg(
            impressions=("event", lambda x: (x == "Loaded a Page").sum()),
            widget_views=("event", lambda x: (x == "Embed Widget Viewed").sum()),
            clicks=("event", lambda x: (x == "Embed Widget Clicked").sum()),
            searches=("event", lambda x: (x == "Travel Searched").sum()),
            unique_users=("anonymous_id", "nunique"),
        )
        .sort_values("impressions", ascending=False)
    )
    # Conversion rates
    result["view_rate"] = (
        (result["widget_views"] / result["impressions"] * 100)
        .where(result["impressions"] > 0, 0)
        .round(1)
    )
    result["click_rate"] = (
        (result["clicks"] / result["impressions"] * 100)
        .where(result["impressions"] > 0, 0)
        .round(2)
    )
    result["search_rate"] = (
        (result["searches"] / result["impressions"] * 100)
        .where(result["impressions"] > 0, 0)
        .round(2)
    )
    return result


def engaged_users(df: pd.DataFrame) -> pd.DataFrame:
    """Find users who viewed or interacted with the travel widget.

    Returns one row per anonymous_id with their engagement summary.
    """
    engagement_events = [
        "Embed Widget Viewed",
        "Embed Widget Clicked",
        "Travel Searched",
        "Property Added to Cart",
        "Checkout Clicked",
        "Search Refined",
        "No Availability Viewed",
    ]
    engaged = df[df["event"].isin(engagement_events)].copy()

    if engaged.empty:
        return pd.DataFrame()

    summary = engaged.groupby("anonymous_id").agg(
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        total_events=("event", "size"),
        widget_views=("event", lambda x: (x == "Embed Widget Viewed").sum()),
        widget_clicks=("event", lambda x: (x == "Embed Widget Clicked").sum()),
        searches=("event", lambda x: (x == "Travel Searched").sum()),
        cart_adds=(
            "event",
            lambda x: (x == "Property Added to Cart").sum(),
        ),
        checkouts=("event", lambda x: (x == "Checkout Clicked").sum()),
        sessions=("session_id", "nunique"),
        utm_sources=("utm_source", lambda x: ", ".join(x.dropna().unique())),
        referrers=(
            "referring_domain",
            lambda x: ", ".join(x.dropna().unique()),
        ),
        request_ips=("request_ip", lambda x: ", ".join(x.dropna().unique())),
    )
    summary["max_funnel_step"] = summary.apply(_max_funnel_step, axis=1)
    return summary.sort_values("total_events", ascending=False)


def _max_funnel_step(row: pd.Series) -> str:
    if row["checkouts"] > 0:
        return "Checkout Clicked"
    if row["cart_adds"] > 0:
        return "Property Added to Cart"
    if row["searches"] > 0:
        return "Travel Searched"
    if row.get("widget_clicks", 0) > 0:
        return "Embed Widget Clicked"
    if row["widget_views"] > 0:
        return "Embed Widget Viewed"
    return "Other Engagement"


def top_search_destinations(df: pd.DataFrame) -> pd.DataFrame:
    """Most searched destinations."""
    searches = df[df["event"] == "Travel Searched"]
    if searches.empty:
        return pd.DataFrame()
    return (
        searches.groupby("search_text")
        .agg(count=("search_text", "size"), unique_users=("anonymous_id", "nunique"))
        .sort_values("count", ascending=False)
    )


_SOURCE_ALIASES = {
    "www.bikereg.com": "BikeReg",
    "www.skireg.com": "SkiReg",
    "www.runreg.com": "RunReg",
    "www.trireg.com": "TriReg",
    "www.skimag.com": "SKI",
}


def _normalize_source(s: str) -> str:
    """Normalize a traffic source label."""
    if not s or s == "(direct)":
        return s
    # Drop dev / localhost sources
    if s.startswith("dev.") or s.startswith("localhost"):
        return None
    return _SOURCE_ALIASES.get(s, s)


def referrer_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Break down events by traffic source.

    Uses ``referring_domain`` when available, falls back to
    ``utm_source`` for direct/empty referrers.  Normalizes
    domain variants (e.g. www.bikereg.com → BikeReg) and
    excludes dev/localhost traffic.
    """
    tmp = df.copy()
    tmp["traffic_source"] = (
        tmp["referring_domain"]
        .where(tmp["referring_domain"] != "", tmp["utm_source"])
        .fillna("(direct)")
        .map(_normalize_source)
    )
    # Drop rows where source was excluded (dev/localhost)
    tmp = tmp[tmp["traffic_source"].notna()]
    return (
        tmp.groupby("traffic_source")
        .agg(
            events=("event", "size"),
            unique_users=("anonymous_id", "nunique"),
            widget_views=(
                "event",
                lambda x: (x == "Embed Widget Viewed").sum(),
            ),
        )
        .sort_values("events", ascending=False)
    )
