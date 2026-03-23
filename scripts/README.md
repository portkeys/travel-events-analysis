# Automated Dashboard Updates

Daily scheduled job that fetches new event data from S3, regenerates the static HTML dashboard, and pushes changes to GitHub Pages.

## How It Works

1. **Checks AWS SSO** — if the session is expired, sends a macOS notification and exits
2. **Fetches new data** — detects the most recent cached date and fetches from there to today (backfills any gaps automatically). Falls back to a full 30-day fetch if the cache is empty
3. **Prunes old cache** — deletes Parquet files older than 30 days to keep the cache bounded
4. **Generates HTML** — runs `generate_html.py` to rebuild `docs/index.html`
5. **Commits & pushes** — if the dashboard changed, commits and pushes to `main`

## Setup (Step by Step)

### 1. Install dependencies

```bash
uv sync
```

### 2. Seed the cache (first time only)

If you don't already have cached data, fetch the last 30 days:

```bash
aws sso login --profile wen_outside
python -m travel_events load --days 30
```

### 3. Make the script executable

```bash
chmod +x scripts/update-dashboard.sh
```

### 4. Test it manually

```bash
./scripts/update-dashboard.sh
cat logs/update-dashboard.log
```

### 5. Install the launchd job

Symlink the plist into `~/Library/LaunchAgents/` and load it:

```bash
ln -s "$(pwd)/scripts/com.portkeys.travel-dashboard.plist" ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.portkeys.travel-dashboard.plist
```

The job runs immediately on load (`RunAtLoad`) and then every 24 hours.

### 6. Verify it's running

```bash
launchctl list | grep travel-dashboard
# Should show PID or exit status + label
```

## Managing the Job

```bash
# Stop the scheduled job
launchctl unload ~/Library/LaunchAgents/com.portkeys.travel-dashboard.plist

# Restart it
launchctl load ~/Library/LaunchAgents/com.portkeys.travel-dashboard.plist

# Check logs
tail -50 logs/update-dashboard.log
```

## When SSO Expires

The AWS SSO session typically lasts 8–12 hours. When it expires:

1. You'll get a macOS notification: *"AWS SSO expired"*
2. Re-authenticate: `aws sso login --profile wen_outside`
3. The next scheduled run (or manual run) will proceed normally

## Files

| File | Purpose |
|------|---------|
| `update-dashboard.sh` | Main pipeline script |
| `com.portkeys.travel-dashboard.plist` | macOS launchd job definition |
