# NFL longest-touchdown props: data collection

Stage 1 of the project: get the data, describe it, and fix the train/test split. No models yet.

## How to run (Mac or Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_data.py      # ~200 MB of parquet into raw/ (2016 to current season)
python3 scripts/build_datasets.py     # builds data/ tables and splits.csv
```

`raw/` and `data/` are git-ignored; rerun the two scripts to regenerate them.

## Source

**nflverse play-by-play** (the data behind nflfastR), free and updated weekly during the season:
`https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_<YEAR>.parquet`

Downloaded seasons 2016 through 2026 (2026 is in progress, weeks 1 to 5 so far). About 48,000 plays and 267 to 285 games per full season, regular season plus playoffs, 372 columns per play.

Supporting files from the same project:
- `raw/ref/players.parquet`: player ids to position (WR, RB, TE...). Play-by-play has no position column, so this join is required.
- `raw/ref/games.csv`: one row per game with closing spread, total, moneylines, roof, surface, weather.

To refresh (e.g. new 2026 weeks), re-download `pbp_2026.parquet` and rerun `python3 scripts/build_datasets.py`.

## Folder layout

| Path | What it is |
|---|---|
| `raw/pbp_2016.parquet` … `raw/pbp_2026.parquet` | Untouched nflverse play-by-play (≈20 MB/season) |
| `raw/ref/` | players and games reference tables |
| `scripts/build_datasets.py` | Builds everything in `data/` from `raw/` |
| `data/td_plays.parquet` | 14,611 TD plays, one row each |
| `data/game_longest_td.parquet` | 2,840 games, longest TD per game (the game prop target) |
| `data/player_games.parquet` | 54,556 player-games: targets, air yards, longest catch/run, TDs |
| `data/splits.csv` | `game_id` → train / valid / test / live |

## Key play-by-play columns

| Need | Column(s) |
|---|---|
| Game / time | `game_id`, `season`, `week`, `season_type` (REG/POST), `qtr`, `game_seconds_remaining` |
| Teams | `posteam` (offense), `defteam` (defense), `td_team` (team that scored) |
| TD flags | `touchdown`, `pass_touchdown`, `rush_touchdown`, `return_touchdown` |
| Yards | `yards_gained`, `air_yards` (depth of target, on every pass attempt incl. incompletions), `yards_after_catch`, `return_yards`, `fumble_recovery_1_yards` |
| Field position | `yardline_100` (yards from opponent's end zone at the snap) |
| Players | `receiver_player_id/name`, `rusher_player_id/name`, `passer_player_id/name`, `td_player_id/name` |
| Volume | `pass_attempt` (a target when `receiver_player_id` is set), `rush_attempt`, `complete_pass` |
| Game context | `spread_line`, `total_line` (closing lines), `roof`, `surface`, `temp`, `wind`, `score_differential` |

Not in the free data: player tracking (speed, separation) and coverage type. nflverse also has Next Gen Stats weekly summaries and snap counts if we want them later.

## What I built

**`td_plays`** adds `td_kind` (pass, rush, interception_return, special_teams_return, fumble_or_other), `td_yards`, `defense_allowing` (the team scored on), and the scorer's `position`. Every offensive TD scorer matched a position.

**`game_longest_td`** has `longest_td` (all TDs, including returns, which is how game "Longest TD" props are usually graded) and `longest_off_td` (passing and rushing only), plus who scored it and the closing spread/total. 18 games had no TD; books grade those differently, so they're flagged in `no_td_game`.

**`player_games`** is for the target-depth theory: `targets`, `air_yards_sum/mean/max`, `deep_targets` (air yards ≥ 20), `receptions`, `longest_rec`, `carries`, `longest_rush`, and `longest_td`.

## Things worth knowing already

- **A scrimmage TD's length is exactly `yardline_100` at the snap** (true for 100% of passing/rushing TDs). So "long TD" really means "scored from far out", and drive start position matters as much as player.
- TD length by scorer position, 2016–2026 (passing/rushing TDs):

  | Position | TDs | Mean yds | Median yds | Share ≥ 40 yds |
  |---|---|---|---|---|
  | WR | 5,481 | 20.5 | 14 | 17% |
  | RB | 4,979 | 10.8 | 4 | 7% |
  | TE | 2,246 | 12.0 | 8 | 4% |
  | QB | 1,015 | 6.6 | 3 | 2% |

  That's already the first row of the baseline table you described.
- Game-level longest TD: median 37 yards, mean 40.7; 44.7% of games have one of 40.5+ yards (a common sportsbook line).

## Train / test split (picked)

Your idea was to shuffle several years together for training and then test on separate data. Here's how I set it up:

- **Train + validation: 2016–2023, shuffled by game.** All games from those 8 seasons are randomly assigned (seed 42), 80% train (1,742 games) and 20% validation (449 games). The shuffle is **by game, not by play**: plays from the same game are correlated, and splitting them would leak.
- **Test: 2024 and 2025 seasons, held out entirely** (570 games). These games are never seen in training, and they're also later in time, which is how the model would actually be used: trained on the past, betting on the future. A random-play test set from the same years would look better than reality.
- **Live: 2026 season so far** (79 games), for a forward check as the season goes on.

One caveat: rolling player features (e.g. "average air yards over the last 4 games") must only use games before the one being predicted, even inside the training set.

For a truly *independent source* check, I tried to cross-validate test-game TDs against ESPN's API, but this environment's network blocks it. The nflverse data is itself built from the NFL's official feed, so I don't expect discrepancies. If you want a second source anyway, Pro Football Reference box scores would be the one to scrape on your Mac.

## Real prop lines: what's free

- **Game "Longest TD" props: no free historical source I could find.** The Odds API (the most common odds feed) doesn't even list this market.
- **Player "longest reception / longest rush / longest completion" props** exist on [The Odds API](https://the-odds-api.com/sports-odds-data/betting-markets.html) (`player_reception_longest`, `player_rush_longest`, `player_pass_longest_completion`). Historical player props start May 3, 2023, but [historical data is only on paid plans](https://the-odds-api.com/historical-odds-data/) and costs 10 credits per market per game. The free tier (500 credits/month) only gets current lines, so we could start logging 2026 lines going forward for free.
- **Free and already in the data:** closing spread and total for every game (`spread_line`, `total_line`). Total points is a strong proxy for how many TDs, and so how many chances at a long one.

Practical plan: evaluate the model against a fixed line (e.g. 40.5 yards, which goes over in about 45% of games historically) for 2016–2025, and log real lines from The Odds API free tier during 2026.
