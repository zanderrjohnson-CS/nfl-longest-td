"""Baseline "lookup tables" for longest-TD props (the piece-square-table analogue).

Every table is estimated on the TRAIN split only (2016-2023 shuffled games, 80%),
then a simple additive baseline is scored on valid (rest of 2016-2023) and test (2024-25).
Small-sample groups (one defense, one coach) are shrunk toward the league rate:
    shrunk = (long + M * league_rate) / (n + M)
so a defense with 3 long TDs out of 10 does not look like the worst in history.

Run after build_datasets.py:  python3 scripts/baseline.py
Outputs: data/baseline/*.csv and data/baseline/report.md
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "baseline")
os.makedirs(OUT, exist_ok=True)
LONG = 40          # a "long" TD is 40+ yards
LINE = 40.5        # a typical game "longest TD" prop line
M_TD = 40          # shrinkage strength for per-TD rates (in TDs)
M_GAME = 30        # shrinkage strength for per-game rates (in team-games)

td = pd.read_parquet(os.path.join(D, "td_plays.parquet"))
games = pd.read_parquet(os.path.join(D, "game_longest_td.parquet"))
splits = pd.read_csv(os.path.join(D, "splits.csv"))[["game_id", "split"]]
coaches = pd.read_csv(os.path.join(ROOT, "raw", "ref", "games.csv"))[["game_id", "home_coach", "away_coach"]]
games = games.merge(splits, on="game_id").merge(coaches, on="game_id", how="left")
td = td.merge(splits, on="game_id")
td["long"] = td.td_yards >= LONG

# ---- one row per team per game (offense view) -------------------------------
def side(home):
    t, o = ("home", "away") if home else ("away", "home")
    x = games[["game_id", "season", "split", "spread_line", "total_line"]].copy()
    x["team"], x["opp"] = games[f"{t}_team"], games[f"{o}_team"]
    x["coach"], x["opp_coach"] = games[f"{t}_coach"], games[f"{o}_coach"]
    x["team_spread"] = games.spread_line if home else -games.spread_line   # + = team favored by that many
    x["implied_pts"] = (games.total_line + x.team_spread) / 2
    return x
tg = pd.concat([side(True), side(False)], ignore_index=True)
off = td[td.offensive_td | ~td.offensive_td].copy()  # every TD, credited to the scoring team
by = off.groupby(["game_id", "td_team"]).agg(n_td=("play_id", "size"), longest=("td_yards", "max"),
                                            n_long=("long", "sum"))
tg = tg.join(by, on=["game_id", "team"])
tg[["n_td", "n_long"]] = tg[["n_td", "n_long"]].fillna(0)
tg["longest"] = tg.longest.fillna(0)
tg["has_long"] = tg.n_long > 0

train_td = td[td.split == "train"]
train_tg = tg[tg.split == "train"]

def shrink(df, k, n, prior, m):
    return (df[k] + m * prior) / (df[n] + m)

tables = {}

# 1. Position table: how long are TDs by scorer position and play type
o = train_td[train_td.offensive_td]
pos = o.groupby(["pos_simple", "td_kind"]).agg(tds=("play_id", "size"), mean_yds=("td_yards", "mean"),
                                               median_yds=("td_yards", "median"), long_rate=("long", "mean"))
tables["position"] = pos[pos.tds >= 20].sort_values("long_rate", ascending=False).round(3)

# 2. Defense table (team-season): long TDs allowed per game, shrunk
league_pg = train_tg.has_long.mean()
dg = tg[tg.split == "train"].groupby(["season", "opp"]).agg(games=("game_id", "size"), long_games=("has_long", "sum"),
                                                          long_tds=("n_long", "sum"), tds=("n_td", "sum"))
dg["raw_rate"] = dg.long_games / dg.games
dg["shrunk_rate"] = shrink(dg, "long_games", "games", league_pg, M_GAME)
dg.index.names = ["season", "defense"]
tables["defense_season"] = dg.sort_values("shrunk_rate", ascending=False).round(3)

# 3. Spread table: team's pregame spread vs. that team's long TDs
bins = [-30, -10, -6.5, -3, 0, 3, 6.5, 10, 30]
labels = ["dog 10+", "dog 7-10", "dog 3.5-6.5", "dog 0.5-3", "fav 0-3", "fav 3.5-6.5", "fav 7-10", "fav 10+"]
x = train_tg.dropna(subset=["team_spread"]).copy()
x["spread_bucket"] = pd.cut(x.team_spread, bins, labels=labels)
sp = x.groupby("spread_bucket", observed=True).agg(team_games=("game_id", "size"), tds_per_game=("n_td", "mean"),
                                                   p_any_long_td=("has_long", "mean"), longest_mean=("longest", "mean"),
                                                   long_tds=("n_long", "sum"), tds=("n_td", "sum"))
sp["long_share_of_tds"] = sp.long_tds / sp.tds
tables["spread"] = sp.drop(columns=["long_tds", "tds"]).round(3)

# 3b. Game total (over/under) vs. the game's longest TD
gt = games[games.split == "train"].dropna(subset=["total_line"]).copy()
gt["total_bucket"] = pd.cut(gt.total_line, [0, 40, 43, 46, 49, 52, 70])
tables["game_total"] = gt.groupby("total_bucket", observed=True).agg(
    games=("game_id", "size"), mean_longest=("longest_td", "mean"),
    p_over_line=("longest_td", lambda s: (s.fillna(0) > LINE).mean())).round(3)

# 4. Coach tables: head coach of the DEFENSE vs. long TDs allowed (overall and by scorer position)
c = train_tg.groupby("opp_coach").agg(games=("game_id", "size"), long_games=("has_long", "sum"))
c["shrunk_rate"] = shrink(c, "long_games", "games", league_pg, M_GAME)
tables["coach_defense"] = c[c.games >= 16].sort_values("shrunk_rate", ascending=False).round(3)

o2 = o.merge(tg[["game_id", "team", "opp_coach"]], left_on=["game_id", "td_team"], right_on=["game_id", "team"])
o2 = o2[o2.pos_simple.isin(["WR", "RB", "TE"])]
pos_rate = o2.groupby("pos_simple").long.mean()
cp = o2.groupby(["opp_coach", "pos_simple"]).agg(tds_allowed=("play_id", "size"), long=("long", "sum")).reset_index()
cp["league_rate"] = cp.pos_simple.map(pos_rate)
cp["shrunk_rate"] = (cp.long + M_TD * cp.league_rate) / (cp.tds_allowed + M_TD)
cp["vs_league"] = cp.shrunk_rate - cp.league_rate
tables["coach_vs_position"] = cp[cp.tds_allowed >= 15].sort_values("vs_league", ascending=False).round(3)

for k, v in tables.items():
    v.to_csv(os.path.join(OUT, f"{k}.csv"))

# ---- the baseline predictor: P(game longest TD > LINE) -----------------------
# Start from the game-total bucket rate, then nudge by both defenses' long-TD tendency
# (prior season's shrunk rate vs league, so it only uses information available pregame).
dfull = tg.groupby(["season", "opp"]).has_long.agg(["sum", "size"]).reset_index()
dfull["rate"] = (dfull["sum"] + M_GAME * league_pg) / (dfull["size"] + M_GAME)
prev = dfull.assign(season=dfull.season + 1).set_index(["season", "opp"]).rate
tot_rate = tables["game_total"].p_over_line

g = games.dropna(subset=["total_line"]).copy()
g["total_bucket"] = pd.cut(g.total_line, [0, 40, 43, 46, 49, 52, 70])
g["p_total"] = g.total_bucket.map(tot_rate).astype(float)
for s in ["home", "away"]:
    g[f"def_{s}"] = [prev.get((se, t), league_pg) for se, t in zip(g.season, g[f"{s}_team"])]
g["def_adj"] = (g.def_home - league_pg) + (g.def_away - league_pg)
g["p_baseline"] = (g.p_total + g.def_adj).clip(0.02, 0.98)
g["over"] = (g.longest_td.fillna(0) > LINE).astype(int)
p_const = g[g.split == "train"].over.mean()

def scores(df, col):
    p = df[col] if col else pd.Series(p_const, index=df.index)
    brier = ((p - df.over) ** 2).mean()
    ll = -(df.over * np.log(p) + (1 - df.over) * np.log(1 - p)).mean()
    return round(brier, 4), round(ll, 4)

ev = []
for sp_ in ["train", "valid", "test", "live"]:
    d = g[g.split == sp_]
    if len(d) == 0:
        continue
    for name, col in [("constant (train over rate)", None), ("game total only", "p_total"),
                      ("total + defenses", "p_baseline")]:
        b, l = scores(d, col)
        ev.append({"split": sp_, "games": len(d), "model": name, "brier": b, "log_loss": l})
ev = pd.DataFrame(ev)
ev.to_csv(os.path.join(OUT, "eval.csv"), index=False)
g[["game_id", "season", "split", "total_line", "p_total", "def_home", "def_away", "p_baseline", "longest_td", "over"]] \
    .to_csv(os.path.join(OUT, "game_predictions.csv"), index=False)

# ---- report ------------------------------------------------------------------
def md(df, n=None):
    return (df.head(n) if n else df).to_markdown()
rep = f"""# Longest-TD baseline report

Tables fit on the train split only. "Long" = {LONG}+ yards; prop line = {LINE}.
League rate that a team scores at least one long TD in a game: {league_pg:.3f}.

## Position
{md(tables['position'])}

## Spread (team's own pregame spread)
{md(tables['spread'])}

## Game total vs. longest TD
{md(tables['game_total'])}

## Defenses (team-season), most long TDs allowed, shrunk
{md(tables['defense_season'], 10)}

## Defensive head coach, long TDs allowed per game, shrunk (16+ games)
Top 10:
{md(tables['coach_defense'], 10)}

Bottom 10:
{md(tables['coach_defense'].tail(10))}

## Defensive head coach vs. scorer position (15+ TDs allowed), biggest gaps vs. league
{md(tables['coach_vs_position'], 10)}

## Baseline predictor: P(game longest TD > {LINE})
Lower is better for both scores. Brier of the constant model ~0.247 means near coin flip.
{ev.to_markdown(index=False)}
"""
open(os.path.join(OUT, "report.md"), "w").write(rep)
print(rep)
