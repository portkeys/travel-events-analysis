"""Load travel widget events from S3 into DataFrames.

Events are cached locally as Parquet files (one per day) under ``data/cache/``.
Subsequent loads for the same day read from disk instead of hitting S3.
Pass ``force_refresh=True`` to re-fetch a day from S3 even if a cache file exists.
Today's date is always re-fetched by default since new events may still arrive.

S3 downloads are parallelized: files within a day are fetched concurrently,
and multiple days are loaded concurrently as well.
"""

import gzip
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import boto3
import pandas as pd


S3_BUCKET = "outside-rudderstack-prod"
S3_PREFIX = "rudder-logs/35OVW4ziaxPwbNqNO2nlCKaU6lx"

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# Concurrency settings
_FILES_PER_DAY_WORKERS = 10  # parallel S3 GetObject calls within one day
_DAY_WORKERS = 4  # parallel days fetched at once


def _s3_client(profile: str = "wen_outside"):
    session = boto3.Session(profile_name=profile)
    return session.client("s3")


def _date_to_prefix(d: date) -> str:
    return f"{S3_PREFIX}/{d.strftime('%m-%d-%Y')}/"


def _cache_path(d: date) -> Path:
    """Return the local Parquet cache path for a given day."""
    return CACHE_DIR / f"{d.isoformat()}.parquet"


def _save_to_cache(d: date, df: pd.DataFrame) -> None:
    """Persist a day's DataFrame to the local Parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        # Coerce mixed-type object columns to strings so PyArrow doesn't choke
        out = df.copy()
        for col in out.select_dtypes(include=["object"]).columns:
            out[col] = out[col].where(out[col].isna(), out[col].astype(str))
        out.to_parquet(_cache_path(d), index=False)


def _load_from_cache(d: date):
    """Load a day from cache. Returns None if not cached."""
    path = _cache_path(d)
    if path.exists():
        return pd.read_parquet(path)
    return None


def _fetch_one_file(s3, key: str) -> list[dict]:
    """Download and parse a single .json.gz file from S3."""
    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
    raw = gzip.decompress(resp["Body"].read())
    events = []
    for line in raw.decode("utf-8").strip().split("\n"):
        if line:
            events.append(json.loads(line))
    return events


def _fetch_day_from_s3(d: date, profile: str = "wen_outside") -> pd.DataFrame:
    """Fetch all events for a single day from S3, downloading files in parallel."""
    s3 = _s3_client(profile)
    prefix = _date_to_prefix(d)

    # Collect all .json.gz keys for this day
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json.gz"):
                keys.append(obj["Key"])

    if not keys:
        return pd.DataFrame()

    # Download files in parallel
    events = []
    with ThreadPoolExecutor(max_workers=_FILES_PER_DAY_WORKERS) as pool:
        futures = {pool.submit(_fetch_one_file, s3, key): key for key in keys}
        for future in as_completed(futures):
            events.extend(future.result())

    if not events:
        return pd.DataFrame()

    return _normalize_events(events)


def load_day(
    d: date,
    profile: str = "wen_outside",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load all events for a single day.

    Reads from local cache when available.  Fetches from S3 and caches
    the result when the cache is missing or ``force_refresh`` is True.
    """
    if not force_refresh:
        cached = _load_from_cache(d)
        if cached is not None:
            return cached

    df = _fetch_day_from_s3(d, profile)
    _save_to_cache(d, df)
    return df


def _load_single_day(
    d: date, profile: str, force_refresh: bool
) -> tuple[date, pd.DataFrame]:
    """Helper for parallel day loading. Returns (date, dataframe)."""
    df = load_day(d, profile, force_refresh=force_refresh)
    return d, df


def load_date_range(
    start: date,
    end: date,
    profile: str = "wen_outside",
    force_refresh: bool = False,
    progress_callback=None,
) -> pd.DataFrame:
    """Load events for a date range (inclusive).

    Each day is loaded from the local Parquet cache when available.
    Today's date is always re-fetched from S3 since new events may still
    be arriving.  Pass ``force_refresh=True`` to re-fetch every day.

    Days that need S3 fetching are downloaded concurrently (up to
    ``_DAY_WORKERS`` at a time).  Cached days are loaded first (fast),
    then uncached days are fetched in parallel.

    ``progress_callback``, if provided, is called with (completed, total)
    after each day finishes loading.
    """
    today = date.today()

    # Split days into cached vs needs-fetch
    all_days = []
    current = start
    while current <= end:
        all_days.append(current)
        current += timedelta(days=1)

    total = len(all_days)
    completed = 0
    frames = []

    # Phase 1: load cached days instantly
    uncached_days = []
    for d in all_days:
        refresh = force_refresh or d == today
        if not refresh:
            cached = _load_from_cache(d)
            if cached is not None:
                print(f"Loading {d.isoformat()} (cache)")
                frames.append(cached)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                continue
        uncached_days.append((d, refresh))

    # Phase 2: fetch uncached days in parallel
    if uncached_days:
        with ThreadPoolExecutor(max_workers=_DAY_WORKERS) as pool:
            futures = {
                pool.submit(_load_single_day, d, profile, refresh): d
                for d, refresh in uncached_days
            }
            for future in as_completed(futures):
                d, df = future.result()
                print(f"Loading {d.isoformat()} (S3)")
                if not df.empty:
                    frames.append(df)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

    if not frames:
        return pd.DataFrame()
    # Drop columns that are entirely NA in individual frames before concat
    # to avoid FutureWarning about all-NA column dtype handling
    frames = [f.dropna(axis=1, how="all") for f in frames]
    return pd.concat(frames, ignore_index=True)


def _normalize_events(events: list[dict]) -> pd.DataFrame:
    """Flatten RudderStack events into a DataFrame."""
    rows = []
    for evt in events:
        props = evt.get("properties", {})
        ctx = evt.get("context", {})
        campaign = ctx.get("campaign", {})
        page = ctx.get("page", {})
        screen = ctx.get("screen", {})

        row = {
            # Core fields
            "event": evt.get("event"),
            "type": evt.get("type"),
            "anonymous_id": evt.get("anonymousId"),
            "user_id": evt.get("userId") or None,
            "message_id": evt.get("messageId"),
            "timestamp": evt.get("timestamp"),
            "received_at": evt.get("receivedAt"),
            "request_ip": evt.get("request_ip"),
            "session_id": ctx.get("sessionId"),
            "session_start": ctx.get("sessionStart", False),
            # User agent / device
            "user_agent": ctx.get("userAgent"),
            "locale": ctx.get("locale"),
            "timezone": ctx.get("timezone"),
            "screen_width": screen.get("width"),
            "screen_height": screen.get("height"),
            # Page context
            "page_url": page.get("url") or props.get("pageUrl"),
            "page_path": page.get("path") or props.get("pagePath"),
            "page_title": page.get("title") or props.get("pageTitle"),
            "referrer": page.get("referrer"),
            "referring_domain": page.get("referring_domain"),
            # UTM / Campaign
            "utm_source": campaign.get("source") or props.get("utm_source"),
            "utm_campaign": campaign.get("name") or props.get("utm_campaign"),
            "utm_content": campaign.get("content") or props.get("utm_content"),
            "utm_term": campaign.get("term") or props.get("utm_term"),
            # Widget-specific
            "variant": props.get("variant"),
            "has_custom_header": props.get("has_custom_header"),
            "domain": props.get("domain"),
            # Search / booking
            "search_text": props.get("search_text"),
            "arrival_date": props.get("arrival_date"),
            "departure_date": props.get("departure_date"),
            "number_of_adults": props.get("number_of_adults"),
            "number_of_children": props.get("number_of_children"),
            "search_surface": props.get("search_surface"),
            "refinement_type": props.get("refinement_type"),
            # Property / cart
            "property_id": props.get("property_id"),
            "supplier_id": props.get("supplier_id"),
            "product_id": props.get("product_id"),
            "quantity": props.get("quantity"),
            # No availability
            "search_type": props.get("search_type"),
            "exit_type": props.get("exit_type"),
            "time_since_no_availability_ms": props.get(
                "time_since_no_availability_ms"
            ),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    return df
