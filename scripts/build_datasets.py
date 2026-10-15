"""Build clean NFL longest-TD datasets from nflverse play-by-play.

Inputs  (raw/): pbp_YYYY.parquet (nflverse), ref/players.parquet, ref/games.csv
Outputs (data/):
  td_plays.parquet      one row per touchdown play, with TD length, scorer, position, defense
  game_longest_td.parquet  one row per game: longest TD (the game-level prop target)
  player_games.parquet  one row per receiver/rusher per game: targets, air yards, longest gain, TDs
  splits.csv            game_id -> train / valid / test / live
Run: python3 scripts/build_datasets.py   (from nfl-longest-td/)
"""
import glob, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, OUT = os.path.join(ROOT, "raw"), os.path.join(ROOT, "data")
SEED = 42
TRAIN_SEASONS = range(2016, 2024)   # 2016-2023: shuffled by game into train/valid
TEST_SEASONS = (2024, 2025)         # held out entirely
LIVE_SEASON = 2026                  # in progress; forward check only

COLS = ["game_id", "play_id", "season", "week", "season_type", "game_date", "home_team", "away_team",
        "posteam", "defteam", "td_team", "play_type", "qtr", "down", "ydstogo", "yardline_100",
        "yards_gained", "air_yards", "yards_after_catch", "return_yards", "fumble_recovery_1_yards",
        "touchdown", "pass_touchdown", "rush_touchdown", "return_touchdown", "interception", "fumble_lost",
        "td_player_id", "td_player_name", "passer_player_id", "passer_player_name",
        "receiver_player_id", "receiver_player_name", "rusher_player_id", "rusher_player_name",
        "pass_attempt", "rush_attempt", "complete_pass", "qb_scramble", "two_point_attempt", "sack",
        "score_differential", "game_seconds_remaining", "spread_line", "total_line", "roof", "surface",
        "temp", "wind", "desc"]

pbp = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(os.path.join(RAW, "pbp_*.parquet")))],
                ignore_index=True)
players = pd.read_parquet(os.path.join(RAW, "ref", "players.parquet"),
                          columns=["gsis_id", "position", "position_group"]).dropna(subset=["gsis_id"])
pos = players.drop_duplicates("gsis_id").set_index("gsis_id")

# ---- TD plays -------------------------------------------------------------
td = pbp[(pbp.touchdown == 1) & (pbp.two_point_attempt != 1)].copy()
td["td_kind"] = np.select(
    [td.pass_touchdown == 1, td.rush_touchdown == 1,
     td.play_type.isin(["kickoff", "punt"]) & (td.td_team != td.posteam),
     td.interception == 1],
    ["pass", "rush", "special_teams_return", "interception_return"], "fumble_or_other")
# Length of the TD: scrimmage TDs = yards_gained (always equals yardline_100 at snap);
# returns use return_yards, fumble returns use the recovery yards.
td["td_yards"] = np.where(td.td_kind.isin(["pass", "rush"]), td.yards_gained,
                  np.where(td.return_yards.fillna(0) > 0, td.return_yards, td.fumble_recovery_1_yards))
td["td_yards"] = td.td_yards.fillna(0).clip(lower=0)
td["offensive_td"] = td.td_kind.isin(["pass", "rush"])
td["defense_allowing"] = np.where(td.offensive_td, td.defteam, td.posteam)  # team scored upon
td = td.join(pos, on="td_player_id")
td["pos_simple"] = td.position.replace({"FB": "RB", "HB": "RB"}).where(td.offensive_td, "DEF/ST")
td = td.drop(columns=["touchdown", "two_point_attempt"])
td.to_parquet(os.path.join(OUT, "td_plays.parquet"), index=False)

# ---- game-level longest TD (game prop target) -----------------------------
games = pd.read_csv(os.path.join(RAW, "ref", "games.csv"))
games = games[games.season >= 2016]
played = set(pbp.game_id.unique())
games = games[games.game_id.isin(played)]
agg = td.groupby("game_id").agg(
    n_td=("play_id", "size"), longest_td=("td_yards", "max"),
    n_off_td=("offensive_td", "sum"))
agg["longest_off_td"] = td[td.offensive_td].groupby("game_id").td_yards.max()
idx = td.groupby("game_id").td_yards.idxmax()
agg = agg.join(td.loc[idx].set_index("game_id")[["td_kind", "td_player_name", "pos_simple", "td_team"]]
               .add_prefix("longest_"))
g = games[["game_id", "season", "game_type", "week", "gameday", "home_team", "away_team", "home_score", "away_score",
           "spread_line", "total_line", "roof", "surface", "temp", "wind", "div_game"]].merge(
    agg, left_on="game_id", right_index=True, how="left")
g["n_td"] = g.n_td.fillna(0).astype(int)
g["no_td_game"] = g.n_td == 0  # most books grade "no TD" as under / void: check rules per book
g.to_parquet(os.path.join(OUT, "game_longest_td.parquet"), index=False)

# ---- player-game table (target depth etc.) ---------------------------------
sc = pbp[(pbp.play_type.isin(["pass", "run"])) & (pbp.two_point_attempt != 1)]
rec = sc[sc.receiver_player_id.notna() & (sc.pass_attempt == 1)].assign(
    deep=lambda x: x.air_yards >= 20, rec_yds=lambda x: x.yards_gained.where(x.complete_pass == 1))
rg = rec.groupby(["game_id", "season", "week", "posteam", "defteam", "receiver_player_id"]).agg(
    player_name=("receiver_player_name", "first"), targets=("play_id", "size"),
    air_yards_sum=("air_yards", "sum"), air_yards_mean=("air_yards", "mean"), air_yards_max=("air_yards", "max"),
    deep_targets=("deep", "sum"), receptions=("complete_pass", "sum"), rec_yards=("rec_yds", "sum"),
    longest_rec=("rec_yds", "max"), rec_td=("pass_touchdown", "sum")).reset_index().rename(
    columns={"receiver_player_id": "player_id"})
ru = sc[sc.rusher_player_id.notna() & (sc.rush_attempt == 1)]
rr = ru.groupby(["game_id", "season", "week", "posteam", "defteam", "rusher_player_id"]).agg(
    rusher_name=("rusher_player_name", "first"), carries=("play_id", "size"), rush_yards=("yards_gained", "sum"),
    longest_rush=("yards_gained", "max"), rush_td=("rush_touchdown", "sum")).reset_index().rename(
    columns={"rusher_player_id": "player_id"})
pg = rg.merge(rr, on=["game_id", "season", "week", "posteam", "defteam", "player_id"], how="outer")
pg["player_name"] = pg.player_name.fillna(pg.rusher_name)
pg = pg.drop(columns="rusher_name")
for c in ["targets", "deep_targets", "receptions", "rec_yards", "rec_td", "carries", "rush_yards", "rush_td", "air_yards_sum"]:
    pg[c] = pg[c].fillna(0)
ltd = td[td.offensive_td].groupby(["game_id", "td_player_id"]).td_yards.max().rename("longest_td")
pg = pg.join(ltd, on=["game_id", "player_id"]).join(pos, on="player_id")
pg.to_parquet(os.path.join(OUT, "player_games.parquet"), index=False)

# ---- split: by game, never by play -----------------------------------------
rng = np.random.default_rng(SEED)
s = g[["game_id", "season"]].copy()
s["split"] = np.where(s.season.isin(TEST_SEASONS), "test", np.where(s.season == LIVE_SEASON, "live", "train"))
tr = s.split == "train"
s.loc[tr, "split"] = np.where(rng.random(tr.sum()) < 0.2, "valid", "train")
s.to_csv(os.path.join(OUT, "splits.csv"), index=False)

print("td_plays", td.shape, "games", g.shape, "player_games", pg.shape)
print(s.groupby("split").size())
