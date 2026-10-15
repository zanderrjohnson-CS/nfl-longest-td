"""Build the King-of-the-End-Zone modelling table.

One row per CANDIDATE per game: every RB/WR/TE/QB/FB who took an offensive snap
(nflverse snap counts), plus one row per team for its D/ST.  Label `king` = 1 if that
candidate scored the game's longest TD (ties: all tied candidates win).  Promo rules:
rushing, receiving and return TDs count for the player; a QB's passing TD does not;
a defensive/special-teams TD also counts for that team's D/ST.

All features are PREGAME: exponentially weighted means of the player's (or team's)
PREVIOUS games only (shift(1)), carried across seasons.  No fixed yardage cutoffs.
Output: data/kotez_candidates.parquet
"""
import glob, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, D = os.path.join(ROOT, "raw"), os.path.join(ROOT, "data")
HALFLIFE = 6   # games; how fast old games fade in the rolling averages
SKILL = {"RB", "WR", "TE", "QB", "FB"}

pbp_cols = ["game_id", "play_id", "season", "week", "posteam", "defteam", "play_type", "yards_gained",
            "air_yards", "yards_after_catch", "complete_pass", "pass_attempt", "rush_attempt", "two_point_attempt",
            "receiver_player_id", "rusher_player_id", "touchdown", "td_team", "td_player_id"]
pbp = pd.concat([pd.read_parquet(f, columns=pbp_cols) for f in sorted(glob.glob(os.path.join(RAW, "pbp_*.parquet")))],
                ignore_index=True)
pbp = pbp[pbp.two_point_attempt != 1]
td = pd.read_parquet(os.path.join(D, "td_plays.parquet"))
games = pd.read_parquet(os.path.join(D, "game_longest_td.parquet"))
sched = pd.read_csv(os.path.join(RAW, "ref", "games.csv"))[["game_id", "gameday", "home_coach", "away_coach"]]
games = games.merge(sched[["game_id", "home_coach", "away_coach"]], on="game_id", how="left")
players = pd.read_parquet(os.path.join(RAW, "ref", "players.parquet"), columns=["gsis_id", "pfr_id", "position"])

# ---- order of games (for "previous games only") ------------------------------
games = games.merge(sched[["game_id"]], on="game_id")
games["order"] = games.season * 100 + games.week

# ---- candidates from snap counts --------------------------------------------
snaps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(RAW, "snaps", "*.parquet")))],
                  ignore_index=True)
pfr2gsis = players.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
snaps["player_id"] = snaps.pfr_player_id.map(pfr2gsis)
snaps = snaps[snaps.position.isin(SKILL) & (snaps.offense_snaps > 0) & snaps.player_id.notna()]
snaps = snaps[snaps.game_id.isin(games.game_id)]
cand = snaps[["game_id", "team", "player_id", "position", "offense_pct"]].rename(columns={"offense_pct": "snap_pct"})
cand["position"] = cand.position.replace({"FB": "RB"})

# ---- per player-game box stats (used only as history) ------------------------
sc = pbp[pbp.play_type.isin(["pass", "run"])]
tgt = sc[sc.receiver_player_id.notna() & (sc.pass_attempt == 1)]
rec = tgt.groupby(["game_id", "receiver_player_id"]).agg(
    targets=("play_id", "size"), air_yards=("air_yards", "sum"),
    receptions=("complete_pass", "sum"),
    yac=("yards_after_catch", "sum"),
    max_rec=("yards_gained", lambda s: s[tgt.loc[s.index, "complete_pass"] == 1].max()))
rec.index.names = ["game_id", "player_id"]
ru = sc[sc.rusher_player_id.notna() & (sc.rush_attempt == 1)]
rus = ru.groupby(["game_id", "rusher_player_id"]).agg(carries=("play_id", "size"), rush_yds=("yards_gained", "sum"),
                                                     max_rush=("yards_gained", "max"))
rus.index.names = ["game_id", "player_id"]
ptd = td[td.td_player_id.notna()].groupby(["game_id", "td_player_id"]).agg(tds=("play_id", "size"),
                                                                          td_len_max=("td_yards", "max"))
ptd.index.names = ["game_id", "player_id"]
box = cand.join(rec, on=["game_id", "player_id"]).join(rus, on=["game_id", "player_id"]).join(ptd, on=["game_id", "player_id"])
for c in ["targets", "air_yards", "receptions", "yac", "carries", "rush_yds", "tds"]:
    box[c] = box[c].fillna(0)
box["max_gain"] = box[["max_rec", "max_rush"]].max(axis=1).fillna(0)
box = box.merge(games[["game_id", "order"]], on="game_id")

# ---- player history features (EWM of previous games) --------------------------
box = box.sort_values(["player_id", "order"])
g = box.groupby("player_id", sort=False)
def hist(col):
    return g[col].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE, ignore_na=True).mean())
box["h_snap_pct"] = hist("snap_pct")
box["h_targets"] = hist("targets")
box["h_carries"] = hist("carries")
box["h_air_yards"] = hist("air_yards")
box["h_max_gain"] = hist("max_gain")
box["h_tds"] = hist("tds")
# ratio features: ratio of weighted sums so low-volume games don't dominate
for num, den, name in [("air_yards", "targets", "h_adot"), ("yac", "receptions", "h_yac_per_rec"),
                       ("rush_yds", "carries", "h_ypc")]:
    n = g[num].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE).mean())
    d_ = g[den].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE).mean())
    box[name] = n / d_.replace(0, np.nan)
box["h_td_len"] = g["td_len_max"].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE, ignore_na=True).mean())
box["h_games"] = g.cumcount()

# ---- team / opponent context ---------------------------------------------------
def side(home):
    t, o = ("home", "away") if home else ("away", "home")
    x = games[["game_id", "season", "week", "order", "spread_line", "total_line"]].copy()
    x["team"], x["opp"] = games[f"{t}_team"], games[f"{o}_team"]
    x["opp_coach"] = games[f"{o}_coach"]
    x["team_spread"] = games.spread_line if home else -games.spread_line
    x["implied_pts"] = (games.total_line + x.team_spread) / 2
    x["home"] = int(home)
    return x.drop(columns="spread_line")
tg = pd.concat([side(True), side(False)], ignore_index=True)

# what each defense allowed, per game (all plays from scrimmage + every TD scored on it)
allowed = sc.groupby(["game_id", "defteam"]).agg(ypp_allowed=("yards_gained", "mean"),
                                                 max_gain_allowed=("yards_gained", "max"))
allowed.index.names = ["game_id", "opp"]
tdl = td.groupby(["game_id", "defense_allowing"]).agg(tds_allowed=("play_id", "size"),
                                                      td_len_allowed=("td_yards", "max"))
tdl.index.names = ["game_id", "opp"]
# D/ST: defensive + return TDs scored by each team
dst = td[~td.offensive_td].groupby(["game_id", "td_team"]).agg(dst_tds=("play_id", "size"))
dst.index.names = ["game_id", "team"]
tg = tg.join(allowed, on=["game_id", "opp"]).join(tdl, on=["game_id", "opp"]).join(dst, on=["game_id", "team"])
tg[["tds_allowed", "dst_tds"]] = tg[["tds_allowed", "dst_tds"]].fillna(0)

def team_hist(df, key, cols, prefix):
    df = df.sort_values([key, "order"])
    gg = df.groupby(key, sort=False)
    for c in cols:
        df[prefix + c] = gg[c].transform(lambda s: s.shift(1).ewm(halflife=HALFLIFE * 2, ignore_na=True).mean())
    return df
# opponent defense history: keyed on the defense (opp), using games where it was the defense
dhist = team_hist(tg[["game_id", "order", "opp", "ypp_allowed", "max_gain_allowed", "tds_allowed", "td_len_allowed"]].copy(),
                  "opp", ["ypp_allowed", "max_gain_allowed", "tds_allowed", "td_len_allowed"], "h_opp_")
chist = team_hist(tg[["game_id", "order", "opp_coach", "max_gain_allowed", "td_len_allowed"]].copy(),
                  "opp_coach", ["max_gain_allowed", "td_len_allowed"], "h_coach_")
thist = team_hist(tg[["game_id", "order", "team", "dst_tds"]].copy(), "team", ["dst_tds"], "h_team_")
tg = tg.merge(dhist.filter(regex="^(game_id|opp|h_)"), on=["game_id", "opp"]) \
       .merge(chist.filter(regex="^(game_id|opp_coach|h_)"), on=["game_id", "opp_coach"]) \
       .merge(thist.filter(regex="^(game_id|team|h_)"), on=["game_id", "team"])

# ---- assemble candidates + D/ST rows --------------------------------------------
keep = ["game_id", "team", "player_id", "position", "h_games", "h_snap_pct", "h_targets", "h_carries", "h_air_yards",
        "h_adot", "h_yac_per_rec", "h_ypc", "h_max_gain", "h_tds", "h_td_len"]
pl = box[keep]
dst_rows = tg[["game_id", "team"]].assign(player_id=lambda x: x.team + "_DST", position="DST")
cands = pd.concat([pl, dst_rows], ignore_index=True).merge(
    tg.drop(columns=["ypp_allowed", "max_gain_allowed", "tds_allowed", "td_len_allowed", "dst_tds"]),
    on=["game_id", "team"])

# ---- label: who was king ---------------------------------------------------------
longest = td.groupby("game_id").td_yards.transform("max")
kings = td[td.td_yards == longest]
winners = set()
for r in kings.itertuples():
    offense_scored = r.td_kind in ("pass", "rush") or (r.play_type in ("pass", "run") and r.td_team == r.posteam)
    if offense_scored:  # incl. an offensive player recovering a teammate's fumble in the end zone
        winners.add((r.game_id, r.td_player_id))
    else:  # return / defensive TD: the team's D/ST wins, and so does the player if he's a candidate
        winners.add((r.game_id, f"{r.td_team}_DST"))
        if isinstance(r.td_player_id, str):
            winners.add((r.game_id, r.td_player_id))
cands["king"] = [int((a, b) in winners) for a, b in zip(cands.game_id, cands.player_id)]
# games where the king isn't in the candidate set (e.g. a QB scoring a long run with no snap row) are kept;
# the model just can't give him credit.  Games with no TD at all have no king and are dropped.
has_king = cands.groupby("game_id").king.transform("max") == 1
cov = has_king.groupby(cands.game_id).first().mean()
cands = cands[has_king]
splits = pd.read_csv(os.path.join(D, "splits.csv"))[["game_id", "split"]]
cands = cands.merge(splits, on="game_id")
cands.to_parquet(os.path.join(D, "kotez_candidates.parquet"), index=False)
print(f"candidates {len(cands):,} rows, {cands.game_id.nunique():,} games; king found in candidate set for {cov:.1%} of games")
print(cands.groupby("split").game_id.nunique())
