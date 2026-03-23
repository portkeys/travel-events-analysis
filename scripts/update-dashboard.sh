#!/bin/bash
set -euo pipefail

PROJECT_DIR="/Users/wyang/Projects/2026/travel-events-analysis"
PROFILE="wen_outside"
DASHBOARD_DAYS=30   # how many days the dashboard covers
LOG_FILE="$PROJECT_DIR/logs/update-dashboard.log"

# Logging setup (append, with timestamps)
mkdir -p "$PROJECT_DIR/logs"
exec >> "$LOG_FILE" 2>&1
echo "=== $(date) ==="

# Step 0: Check SSO session is valid
if ! aws sts get-caller-identity --profile "$PROFILE" > /dev/null 2>&1; then
    echo "ERROR: AWS SSO session expired. Run: aws sso login --profile $PROFILE"
    osascript -e 'display notification "AWS SSO expired. Run: aws sso login --profile wen_outside" with title "Dashboard Update Failed"'
    exit 1
fi

# Step 1: Activate venv, fetch missing days from S3
cd "$PROJECT_DIR"
source .venv/bin/activate

# Determine how many days to fetch: from the most recent cache file to today,
# plus 1 extra day to cover late-arriving S3 data. Falls back to full
# DASHBOARD_DAYS if the cache is empty.
CACHE_DIR="$PROJECT_DIR/data/cache"
if [ -d "$CACHE_DIR" ] && ls "$CACHE_DIR"/*.parquet 1>/dev/null 2>&1; then
    LATEST_CACHE=$(ls "$CACHE_DIR"/*.parquet | sort | tail -1 | xargs basename | sed 's/\.parquet//')
    DAYS_SINCE=$(( ($(date +%s) - $(date -j -f "%Y-%m-%d" "$LATEST_CACHE" +%s)) / 86400 + 1 ))
    # Clamp to at least 2 (today + yesterday for late data)
    FETCH_DAYS=$(( DAYS_SINCE > 2 ? DAYS_SINCE : 2 ))
    echo "Last cached date: $LATEST_CACHE, fetching $FETCH_DAYS days"
else
    FETCH_DAYS=$DASHBOARD_DAYS
    echo "No cache found, fetching full $FETCH_DAYS days"
fi

python -m travel_events load --days "$FETCH_DAYS" --profile "$PROFILE"

# Step 2: Prune cache files older than DASHBOARD_DAYS
find "$PROJECT_DIR/data/cache" -name "*.parquet" -mtime +${DASHBOARD_DAYS} -delete
echo "Pruned parquet files older than $DASHBOARD_DAYS days"

# Step 3: Generate HTML
python generate_html.py

# Step 4: Commit & push (skip if nothing changed)
git -C "$PROJECT_DIR" add docs/index.html
git -C "$PROJECT_DIR" diff --cached --quiet || {
    git -C "$PROJECT_DIR" commit -m "Auto-update dashboard $(date +%Y-%m-%d)"
    git -C "$PROJECT_DIR" push origin main
    echo "Pushed updated dashboard"
}

echo "Done"
