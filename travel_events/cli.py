"""Command-line interface for loading travel event data from S3."""

import argparse
import sys
from datetime import date, timedelta


def _parse_date(s: str) -> date:
    """Parse an ISO-format date string (YYYY-MM-DD)."""
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {s!r} (expected YYYY-MM-DD)")


def _progress(completed: int, total: int) -> None:
    """Print a simple progress indicator."""
    print(f"  [{completed}/{total}] days loaded", end="\r", flush=True)


def _print_summary(df) -> None:
    """Print a summary of the loaded DataFrame."""
    import pandas as pd

    if df.empty:
        print("\nNo events found for the specified date range.")
        return

    total_events = len(df)
    unique_visitors = df["anonymous_id"].nunique()

    # Determine date range from timestamps
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        ts = df["timestamp"].dropna()
        min_date = ts.min().strftime("%Y-%m-%d")
        max_date = ts.max().strftime("%Y-%m-%d")
    else:
        min_date = max_date = "unknown"

    print()
    print("=" * 50)
    print("  Load Summary")
    print("=" * 50)
    print(f"  Total events:     {total_events:,}")
    print(f"  Unique visitors:  {unique_visitors:,}")
    print(f"  Date range:       {min_date} to {max_date}")

    # Events by type
    if "type" in df.columns:
        print()
        print("  Events by type:")
        type_counts = df["type"].value_counts()
        for event_type, count in type_counts.items():
            print(f"    {event_type:<20s} {count:>10,}")

    # Events by event name (top 15)
    if "event" in df.columns:
        named = df[df["event"].notna()]
        if not named.empty:
            print()
            print("  Top events:")
            event_counts = named["event"].value_counts().head(15)
            for event_name, count in event_counts.items():
                print(f"    {event_name:<40s} {count:>10,}")

    print("=" * 50)


def cmd_load(args: argparse.Namespace) -> None:
    """Execute the 'load' sub-command."""
    from travel_events.loader import load_date_range

    # Resolve start/end dates
    if args.days is not None:
        end = date.today()
        start = end - timedelta(days=args.days)
    elif args.start is not None and args.end is not None:
        start = args.start
        end = args.end
    else:
        print("Error: provide either --start and --end, or --days.", file=sys.stderr)
        sys.exit(1)

    if start > end:
        print(f"Error: start date ({start}) is after end date ({end}).", file=sys.stderr)
        sys.exit(1)

    num_days = (end - start).days + 1
    print(f"Loading events from {start} to {end} ({num_days} days)")
    print(f"  AWS profile: {args.profile}")
    print(f"  Force refresh: {args.force_refresh}")
    print()

    df = load_date_range(
        start=start,
        end=end,
        profile=args.profile,
        force_refresh=args.force_refresh,
        progress_callback=_progress,
    )

    # Clear the progress line
    print(" " * 40, end="\r")

    _print_summary(df)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="travel_events",
        description="Load and analyze travel widget events from S3.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- load sub-command ---
    load_parser = subparsers.add_parser(
        "load", help="Load and cache event data from S3"
    )

    date_group = load_parser.add_argument_group("date range")
    date_group.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="Start date (YYYY-MM-DD), inclusive",
    )
    date_group.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="End date (YYYY-MM-DD), inclusive",
    )
    date_group.add_argument(
        "--days",
        type=int,
        default=None,
        help="Load the last N days (alternative to --start/--end)",
    )

    load_parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Re-fetch from S3 even if cached locally",
    )
    load_parser.add_argument(
        "--profile",
        type=str,
        default="wen_outside",
        help="AWS profile name (default: wen_outside)",
    )

    load_parser.set_defaults(func=cmd_load)

    return parser


def main(argv=None) -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
