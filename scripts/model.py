"""King of the End Zone: P(candidate scores the game's longest TD).

Model: conditional logit (softmax over every candidate in the same game), so the
probabilities in each game sum to 1 and the model learns who beats whom.
Features are added in small groups.  Inside each group every feature is tried one at a
time (forward selection) and kept only if it lowers the 5-fold cross-validated log loss on
2016-23 (train + valid games, folds split by game) by more than MIN_GAIN.  Averaging over
5 folds stops a feature from sneaking in on one lucky split.  The final model is refit on
all 2016-23 games; the test seasons (2024-25) are touched once, at the end.
A LightGBM version on the selected features is compared at the end to check for
non-linear effects the logit can't see.

Run after features.py:  python3 scripts/model.py
Outputs: data/model/{stepwise.csv, coefficients.csv, eval.csv, test_predictions.parquet, report.md}
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "model")
os.makedirs(OUT, exist_ok=True)
L2 = 1.0          # ridge penalty on coefficients
MIN_GAIN = 0.001   # required drop in CV log loss (nats/game) to keep a feature
FOLDS = 5

c = pd.read_parquet(os.path.join(D, "kotez_candidates.parquet"))
c = c.sort_values(["game_id", "team", "player_id"]).reset_index(drop=True)

# ---- feature engineering (no thresholds: logs and missing flags only) --------------
for p in ["WR", "TE", "RB", "QB", "DST"]:
    c[f"pos_{p}"] = (c.position == p).astype(float)
c["no_history"] = ((c.h_games.fillna(0) == 0) & (c.position != "DST")).astype(float)
c["log_h_games"] = np.log1p(c.h_games.fillna(0))
c["is_receiver"] = c.position.isin(["WR", "TE"]).astype(float)
train_mask = c.split == "train"
num = [x for x in c.columns if x.startswith("h_") and x != "h_games"] + ["implied_pts", "team_spread", "total_line"]
for col in num:  # fill gaps with the position's training median, then standardize on train
    med = c[train_mask].groupby("position")[col].median()
    c[col] = c[col].fillna(c.position.map(med)).fillna(c.loc[train_mask, col].median())
# position-specific versions of opponent/context features (a pass defense matters to WRs, not RBs)
for col in ["h_opp_ypp_allowed", "h_opp_max_gain_allowed", "h_opp_td_len_allowed", "h_coach_max_gain_allowed",
            "h_coach_td_len_allowed", "implied_pts"]:
    c[col + "_x_rec"] = c[col] * c.is_receiver
num += [x for x in c.columns if x.endswith("_x_rec")]
for col in num + ["log_h_games"]:
    mu, sd = c.loc[train_mask, col].mean(), c.loc[train_mask, col].std()
    c[col] = (c[col] - mu) / (sd if sd > 0 else 1)

GROUPS = [
    ("1 position", ["pos_TE", "pos_RB", "pos_QB", "pos_DST"]),           # WR is the reference
    ("2 usage", ["h_snap_pct", "h_targets", "h_carries", "log_h_games", "no_history"]),
    ("3 target depth", ["h_adot", "h_air_yards", "h_yac_per_rec", "h_ypc"]),
    ("4 big-play / TD history", ["h_max_gain", "h_tds", "h_td_len", "h_team_dst_tds"]),
    ("5 spread & total", ["implied_pts", "implied_pts_x_rec", "team_spread", "total_line", "home"]),
    ("6 opponent defense", ["h_opp_ypp_allowed", "h_opp_max_gain_allowed", "h_opp_tds_allowed", "h_opp_td_len_allowed",
                            "h_opp_ypp_allowed_x_rec", "h_opp_max_gain_allowed_x_rec", "h_opp_td_len_allowed_x_rec"]),
    ("7 opponent head coach", ["h_coach_max_gain_allowed", "h_coach_td_len_allowed",
                               "h_coach_max_gain_allowed_x_rec", "h_coach_td_len_allowed_x_rec"]),
]

# ---- conditional logit ----------------------------------------------------------
class Data:
    def __init__(self, df):
        self.df = df
        codes, self.starts = np.unique(df.game_id.values, return_index=True)
        self.gidx = np.repeat(np.arange(len(self.starts)), np.diff(np.append(self.starts, len(df))))
        self.y = df.king.values.astype(float)
        self.n_games = len(self.starts)

def seg_logsumexp(s, d, mask=None):
    m = np.maximum.reduceat(s, d.starts)
    e = np.exp(s - m[d.gidx])
    if mask is not None:
        e = e * mask
    return np.log(np.add.reduceat(e, d.starts)) + m

def fit(d, feats):
    X = d.df[feats].values
    def f(w):
        s = X @ w
        lse_all, lse_win = seg_logsumexp(s, d), seg_logsumexp(s, d, d.y)
        p = np.exp(s - lse_all[d.gidx])
        q = d.y * np.exp(s - lse_win[d.gidx])
        loss = (lse_all - lse_win).sum() + L2 * w @ w
        grad = X.T @ (p - q) + 2 * L2 * w
        return loss, grad
    w = minimize(f, np.zeros(len(feats)), jac=True, method="L-BFGS-B").x
    return w

def predict(d, feats, w):
    s = d.df[feats].values @ w
    return np.exp(s - seg_logsumexp(s, d)[d.gidx])

def nll(d, p):  # mean over games of -log P(the actual king)
    return float(-np.log(np.add.reduceat(p * d.y, d.starts) + 1e-12).mean())

tr, va, te, lv = (Data(c[c.split == s].reset_index(drop=True)) for s in ["train", "valid", "test", "live"])
dev_df = c[c.split.isin(["train", "valid"])].reset_index(drop=True)
gid = dev_df.game_id.unique()
fold_of = dict(zip(gid, np.random.default_rng(0).permutation(len(gid)) % FOLDS))
dev_df["fold"] = dev_df.game_id.map(fold_of)
cv = [(Data(dev_df[dev_df.fold != k].reset_index(drop=True)), Data(dev_df[dev_df.fold == k].reset_index(drop=True)))
      for k in range(FOLDS)]
dev = Data(dev_df)

def cv_nll(feats):
    if not feats:
        return float(np.mean([nll(v, uniform(v)) for _, v in cv]))
    return float(np.mean([nll(v, predict(v, feats, fit(t, feats))) for t, v in cv]))
uniform = lambda d: np.repeat(1 / np.diff(np.append(d.starts, len(d.df))), np.diff(np.append(d.starts, len(d.df))))

# ---- forward selection, group by group ---------------------------------------------
rows, selected = [], []
best = cv_nll([])
rows.append({"group": "0 none (uniform)", "feature": "-", "cv_nll": round(best, 4), "kept": True})
for gname, feats in GROUPS:
    remaining = list(feats)
    while remaining:
        trial = {}
        for f in remaining:
            trial[f] = cv_nll(selected + [f])
        f_best = min(trial, key=trial.get)
        for f in remaining:
            rows.append({"group": gname, "feature": f, "cv_nll": round(trial[f], 4),
                         "gain_vs_current": round(best - trial[f], 4), "kept": False})
        if best - trial[f_best] > MIN_GAIN:
            selected.append(f_best)
            best = trial[f_best]
            remaining.remove(f_best)
            rows[-len(trial) + list(trial).index(f_best)]["kept"] = True
        else:
            break
    print(f"{gname:28s} CV NLL {best:.4f}  features: {len(selected)}")
step = pd.DataFrame(rows)
step.to_csv(os.path.join(OUT, "stepwise.csv"), index=False)

# ---- final fit on all 2016-23 games, evaluate --------------------------------------------
w = fit(dev, selected)
coef = pd.DataFrame({"feature": selected, "coef_per_sd": w}).sort_values("coef_per_sd", key=abs, ascending=False)
coef.to_csv(os.path.join(OUT, "coefficients.csv"), index=False)
pos_feats = GROUPS[0][1]
w_pos = fit(dev, pos_feats)

import lightgbm as lgb
gbm_feats = sorted(set(selected) | set(pos_feats))
gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
gbm.fit(dev.df[gbm_feats], dev.y)
def gbm_pred(d):
    r = gbm.predict_proba(d.df[gbm_feats])[:, 1]
    return r / np.add.reduceat(r, d.starts)[d.gidx]

ev = []
for name, d in [("test 2024-25", te), ("live 2026", lv)]:
    preds = {"uniform (every candidate equal)": uniform(d), "position only": predict(d, pos_feats, w_pos),
             "conditional logit (selected)": predict(d, selected, w), "LightGBM (selected)": gbm_pred(d)}
    for m, p in preds.items():
        top = pd.Series(p).groupby(d.gidx).idxmax().values
        ev.append({"split": name, "games": d.n_games, "model": m, "nll": round(nll(d, p), 4),
                   "p_actual_king": round(float(np.exp(-nll(d, p))), 4),
                   "top_pick_hit_rate": round(float(d.y[top].mean()), 3)})
ev = pd.DataFrame(ev)
ev.to_csv(os.path.join(OUT, "eval.csv"), index=False)

# calibration on test: do 10% predictions win 10% of the time?
pt = predict(te, selected, w)
cal = pd.DataFrame({"p": pt, "y": te.y})
cal["bin"] = pd.qcut(cal.p, 10, duplicates="drop")
cal = cal.groupby("bin", observed=True).agg(candidates=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean")).round(4)
cal.to_csv(os.path.join(OUT, "calibration_test.csv"))
out = te.df[["game_id", "team", "player_id", "position", "king"]].assign(p_king=pt, p_king_gbm=gbm_pred(te))
out.to_parquet(os.path.join(OUT, "test_predictions.parquet"), index=False)

# value of picking the model's favourite vs. the position-only favourite, by position of the pick
pos_share = te.df.assign(p=pt).groupby("position").agg(model_prob=("p", "sum"), actual_kings=("king", "sum"))
pos_share = (pos_share / te.n_games).round(3)

rep = f"""# King of the End Zone model

Target: which candidate (any RB/WR/TE/QB who took a snap, plus each team's D/ST) scores the game's longest TD.
Model: conditional logit over all candidates in a game, features added a few at a time, kept only if
5-fold cross-validated log loss on 2016-23 games drops by more than {MIN_GAIN}.  Final fit on all 2016-23 games;
test = 2024-25 (untouched until the end), live = 2026 so far.

## Forward selection (5-fold CV log loss per game on 2016-23, lower is better)
{step.to_markdown(index=False)}

## Selected features (standardized coefficients; + means more likely to be king)
{coef.round(3).to_markdown(index=False)}

## Evaluation
`p_actual_king` = geometric-mean probability the model gave the real king. `top_pick_hit_rate` = how often the
model's single favourite was the king.
{ev.to_markdown(index=False)}

## Calibration on test (logit)
{cal.to_markdown()}

## Share of kings by position on test (per game): model expectation vs. actual
{pos_share.to_markdown()}
"""
open(os.path.join(OUT, "report.md"), "w").write(rep)
print(ev.to_string(index=False))
print(coef.round(3).to_string(index=False))
