"""Download nflverse play-by-play and reference tables into raw/.

Run: python3 scripts/download_data.py [first_season] [last_season]   (default 2016 to current)
Re-run any time to refresh the in-progress season.
"""
import datetime, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
PBP = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
REF = {
    "players.parquet": "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet",
    "games.csv": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
}

today = datetime.date.today()
first = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
last = int(sys.argv[2]) if len(sys.argv) > 2 else (today.year if today.month >= 9 else today.year - 1)
os.makedirs(os.path.join(RAW, "ref"), exist_ok=True)

for year in range(first, last + 1):
    path = os.path.join(RAW, f"pbp_{year}.parquet")
    print("downloading", year)
    urllib.request.urlretrieve(PBP.format(year), path)
for name, url in REF.items():
    print("downloading", name)
    urllib.request.urlretrieve(url, os.path.join(RAW, "ref", name))
