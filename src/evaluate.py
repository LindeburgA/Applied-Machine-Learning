"""Rolling-origin evaluation (D11) and decision metrics (D13)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, brier_score_loss
from sklearn.model_selection import ParameterGrid

from . import config
from .features import FEATURES
from .models import MODELS


def modelling_data(df):
    """Only players who stayed have a target (D7)."""
    return df[(df["left_club"] == "stayed") & df["target"].notna()].copy()


def _tune(model, grid, train, val):
    best, best_score = None, -np.inf
    for params in ParameterGrid(grid):
        m = clone(model).set_params(**params).fit(train[FEATURES], train["target"])
        score = average_precision_score(val["target"], m.predict_proba(val[FEATURES])[:, 1])
        if score > best_score:
            best, best_score = params, score
    return best, best_score


def rolling_predictions(df, train_leagues, test_seasons, importance=True):
    data = modelling_data(df)
    dk = data["league"] == config.TARGET_LEAGUE
    in_train = data["league"].isin(train_leagues)
    preds, params_log, importances = [], [], []

    for t in test_seasons:
        test = data[dk & (data["season"] == t)]
        train = data[in_train & (data["season"] < t)]
        inner_train = data[in_train & (data["season"] < t - 1)]
        val = data[dk & (data["season"] == t - 1)]           # tune on the Danish season before t

        out = test[["season", "player_id", "club_id", "target", "share_prev", "age", "position"]].copy()
        out["p_baseline"] = test["share_prev"]                # D12 baseline: last season's share
        for name, factory in MODELS.items():
            model, grid = factory()
            params, val_score = _tune(model, grid, inner_train, val)
            fitted = clone(model).set_params(**params).fit(train[FEATURES], train["target"])
            out[f"p_{name}"] = fitted.predict_proba(test[FEATURES])[:, 1]
            params_log.append({"test_season": t, "model": name, "val_pr_auc": round(val_score, 3),
                               "n_train": len(train), **params})
            if importance:
                pi = permutation_importance(fitted, test[FEATURES], test["target"],
                                            scoring="average_precision", n_repeats=10,
                                            random_state=config.RANDOM_STATE)
                importances.append(pd.DataFrame({"model": name, "test_season": t, "feature": FEATURES,
                                                 "importance": pi.importances_mean}))
        preds.append(out)
        print(f"  test season {t}: train {len(train)} rows, test {len(test)} Danish rows")

    imp = pd.concat(importances) if importances else pd.DataFrame()
    return pd.concat(preds, ignore_index=True), pd.DataFrame(params_log), imp


def threshold_for(cost_ratio):
    """Keep only if p >= c / (c + 1), where c = cost of a wrong keep / cost of a wrong loan (D13)."""
    return cost_ratio / (cost_ratio + 1)


def _decision_stats(y, keep):
    tp = int(((keep == 1) & (y == 1)).sum())
    n_keep = int(keep.sum())
    return {"n_keep": n_keep,
            "precision": tp / n_keep if n_keep else np.nan,
            "recall": tp / int(y.sum()) if y.sum() else np.nan}


def _bootstrap_ci(y, score, fn, n=config.N_BOOTSTRAP):
    rng = np.random.default_rng(config.RANDOM_STATE)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(fn(y[i], score[i]))
    return np.nanpercentile(vals, [2.5, 97.5])


def summarise(preds):
    y = preds["target"].to_numpy()
    fringe = (preds["share_prev"] < config.THRESHOLD).to_numpy()   # the hard cases: below 30% last season
    rows = []
    names = ["baseline"] + list(MODELS)
    for name in names:
        p = preds[f"p_{name}"].to_numpy()
        lo, hi = _bootstrap_ci(y, p, average_precision_score)
        row = {"model": name, "pr_auc": average_precision_score(y, p), "pr_auc_ci": f"[{lo:.2f}, {hi:.2f}]",
               "pr_auc_fringe": average_precision_score(y[fringe], p[fringe])}
        if name == "baseline":
            # the baseline decision rule is "played >= 30% last season"
            row.update({f"{k}@baseline_rule": v for k, v in _decision_stats(y, p >= config.THRESHOLD).items()})
        else:
            row["brier"] = brier_score_loss(y, p)
            for c in config.COST_RATIOS:
                thr = threshold_for(c)
                for k, v in _decision_stats(y, p >= thr).items():
                    row[f"{k}@{c}:1"] = v
            main = threshold_for(config.MAIN_COST_RATIO)
            prec = lambda yy, pp: _decision_stats(yy, pp >= main)["precision"]
            lo, hi = _bootstrap_ci(y, p, prec)
            row[f"precision@{config.MAIN_COST_RATIO}:1_ci"] = f"[{lo:.2f}, {hi:.2f}]"
        rows.append(row)
    overall = pd.DataFrame(rows)

    per_season = (preds.groupby("season")
                  .apply(lambda g: pd.Series({n: average_precision_score(g["target"], g[f"p_{n}"]) for n in names}),
                         include_groups=False)
                  .round(3))
    return overall, per_season


def plots(preds, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["baseline"] + list(MODELS):
        PrecisionRecallDisplay.from_predictions(preds["target"], preds[f"p_{name}"], name=name, ax=ax)
    ax.axhline(preds["target"].mean(), ls="--", c="grey", lw=1, label="random")
    ax.set_title("Precision-recall, Danish test seasons pooled")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in MODELS:
        CalibrationDisplay.from_predictions(preds["target"], preds[f"p_{name}"], n_bins=8,
                                            strategy="quantile", name=name, ax=ax)
    ax.set_title("Calibration, Danish test seasons pooled")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=150)
    plt.close("all")
