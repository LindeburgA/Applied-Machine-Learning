"""Features (D8). Every function only uses rows dated strictly before the snapshot (D9.1)."""
import numpy as np
import pandas as pd

from . import config
from .snapshot import club_game_rows, last_club_assignment, league_games, snapshot_date

NUMERIC = [
    "age", "height_in_cm", "academy_promotion", "cup_squad_only",
    # prior playing time
    "share_prev", "starts_share_prev", "squad_share_prev", "absent_share_prev",
    "share_h1_prev", "share_h2_prev", "share_trend_prev", "cup_minutes_prev", "minutes_all_365",
    # output
    "goals_p90_prev", "assists_p90_prev", "cards_p90_prev",
    # own value
    "log_value", "value_age_days", "value_change_6m", "value_change_12m",
    # competition for the spot
    "n_same_pos", "max_log_value_same_pos", "log_sum_value_same_pos", "n_signings_same_pos",
    "value_rank_in_pos",
    # club context
    "club_ppg_prev", "log_club_squad_value",
]
CATEGORICAL = ["position", "sub_position", "foot", "league"]
FEATURES = NUMERIC + CATEGORICAL


def asof_value(vals, player_ids, dates, max_age_days=config.VALUE_MAX_AGE_DAYS):
    """Latest valuation dated strictly before each date (D9.3). Returns (value, date) aligned to input."""
    left = pd.DataFrame({"player_id": np.asarray(player_ids, dtype="int64"),
                         "asof": pd.to_datetime(np.asarray(dates)).astype(vals["date"].dtype)})
    left["_i"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("asof"), vals.rename(columns={"date": "value_date"}),
                      left_on="asof", right_on="value_date", by="player_id", direction="backward",
                      allow_exact_matches=False, tolerance=pd.Timedelta(days=max_age_days))
    m = m.sort_values("_i")
    return m["market_value_in_eur"].to_numpy(), m["value_date"].to_numpy()


def _prior_playing_time(tables, pop, league, season, snap):
    prev = league_games(tables, league, season - 1, before=snap)
    cg = club_game_rows(prev)
    mid = cg.groupby("club_id")["date"].median().rename("mid")
    cg = cg.join(mid, on="club_id")
    cg["half"] = np.where(cg["date"] <= cg["mid"], 1, 2)
    club_n = cg.groupby("club_id").agg(n=("game_id", "size"), n1=("half", lambda h: (h == 1).sum()),
                                       ppg=("points", "mean"))
    club_n["n2"] = club_n["n"] - club_n["n1"]

    a = tables["apps"]
    a = a[a["game_id"].isin(prev["game_id"])].merge(cg[["game_id", "club_id", "half"]],
                                                    left_on=["game_id", "player_club_id"],
                                                    right_on=["game_id", "club_id"])
    a = a.assign(m_h1=a["minutes_played"].where(a["half"] == 1, 0))
    p = a.groupby(["player_id", "club_id"]).agg(
        minutes=("minutes_played", "sum"), m1=("m_h1", "sum"),
        goals=("goals", "sum"), assists=("assists", "sum"),
        cards=("yellow_cards", "sum"), reds=("red_cards", "sum")).reset_index()

    lu = tables["lineups"].merge(cg[["game_id", "club_id"]], on=["game_id", "club_id"])
    lu = lu.groupby(["player_id", "club_id"]).agg(
        squads=("type", "size"), starts=("type", lambda t: (t == "starting_lineup").sum())).reset_index()

    cup = tables["cup_of"].get(league)
    g = tables["games"]
    cup_ids = g[(g["competition_id"] == cup) & (g["season"] == season - 1) & (g["date"] < snap)]["game_id"]
    cupm = (tables["apps"][tables["apps"]["game_id"].isin(cup_ids)]
            .groupby(["player_id", "player_club_id"])["minutes_played"].sum()
            .rename("cup_minutes_prev").reset_index().rename(columns={"player_club_id": "club_id"}))

    out = (pop[["player_id", "club_id"]]
           .merge(p, on=["player_id", "club_id"], how="left")
           .merge(lu, on=["player_id", "club_id"], how="left")
           .merge(cupm, on=["player_id", "club_id"], how="left")
           .join(club_n, on="club_id"))
    for c in ["minutes", "m1", "goals", "assists", "cards", "reds", "squads", "starts", "cup_minutes_prev"]:
        out[c] = out[c].fillna(0)
    n = out["n"]
    out["share_prev"] = out["minutes"] / (n * 90)
    out["starts_share_prev"] = out["starts"] / n
    out["squad_share_prev"] = out["squads"] / n
    out["absent_share_prev"] = 1 - out["squad_share_prev"]
    out["share_h1_prev"] = out["m1"] / (out["n1"] * 90)
    out["share_h2_prev"] = (out["minutes"] - out["m1"]) / (out["n2"] * 90)
    out["share_trend_prev"] = out["share_h2_prev"] - out["share_h1_prev"]
    p90 = out["minutes"].where(out["minutes"] >= 90) / 90
    out["goals_p90_prev"] = out["goals"] / p90
    out["assists_p90_prev"] = out["assists"] / p90
    out["cards_p90_prev"] = (out["cards"] + out["reds"]) / p90
    out["club_ppg_prev"] = out["ppg"]

    # minutes in any club competition in the 365 days before the snapshot
    a365 = tables["apps"]
    a365 = a365[(a365["date"] >= snap - pd.Timedelta(days=365)) & (a365["date"] < snap)
                & a365["player_id"].isin(pop["player_id"])]
    out["minutes_all_365"] = out["player_id"].map(a365.groupby("player_id")["minutes_played"].sum()).fillna(0)
    return out


def _squad_on_snapshot(tables, pop, league, season, snap):
    """All players (any age) at the pop clubs on 1 July: last squad club in s-1 minus known
    departures before the snapshot, plus signings in the 6 months before it (D4b)."""
    clubs = set(pop["club_id"])
    sq = last_club_assignment(tables, league, season)
    sq = sq[sq["club_id"].isin(clubs)][["player_id", "club_id", "last_squad_date"]]
    t = tables["transfers"]
    gone = t.merge(sq, on="player_id")
    gone = gone[(gone["from_club_id"] == gone["club_id"]) & (gone["transfer_date"] > gone["last_squad_date"])
                & (gone["transfer_date"] < snap)]
    sq = sq[~sq["player_id"].isin(gone["player_id"])].assign(signing=0)

    sign = t[t["to_club_id"].isin(clubs) & (t["transfer_date"] < snap)
             & (t["transfer_date"] >= snap - pd.Timedelta(days=183))
             & (t["from_club_id"] != t["to_club_id"])]
    sign = sign.sort_values("transfer_date").groupby("player_id").tail(1)
    sign = sign[~sign["player_id"].isin(sq["player_id"])]
    sq = pd.concat([sq[["player_id", "club_id", "signing"]],
                    pd.DataFrame({"player_id": sign["player_id"], "club_id": sign["to_club_id"], "signing": 1})])
    # make sure every pop player is in the squad (academy promotions have no squad history)
    extra = pop[~pop.set_index(["player_id", "club_id"]).index.isin(sq.set_index(["player_id", "club_id"]).index)]
    sq = pd.concat([sq, extra[["player_id", "club_id"]].assign(signing=0)], ignore_index=True)
    sq = sq.drop_duplicates(["player_id", "club_id"])
    sq = sq.merge(tables["players"][["player_id", "position"]], on="player_id", how="left")
    v, _ = asof_value(tables["vals"], sq["player_id"], np.full(len(sq), snap))
    sq["value"] = v
    return sq


def _competition_for_spot(tables, pop, league, season, snap):
    sq = _squad_on_snapshot(tables, pop, league, season, snap)
    sq["value_f"] = sq["value"].fillna(0)
    sq["log_value_sq"] = np.log1p(sq["value_f"])
    grp = sq.groupby(["club_id", "position"])
    sq["pos_n"] = grp["player_id"].transform("size")
    sq["pos_sum"] = grp["value_f"].transform("sum")
    sq["pos_sign"] = grp["signing"].transform("sum")
    sq["value_rank_in_pos"] = grp["value_f"].rank(ascending=False, method="average") / sq["pos_n"]
    club_val = sq.groupby("club_id")["value_f"].sum()

    # max log value of teammates in the same position, excluding the player himself
    def max_excl(df):
        v = df["log_value_sq"].to_numpy()
        order = np.argsort(-v)
        top1, top2 = v[order[0]], (v[order[1]] if len(v) > 1 else np.nan)
        return pd.Series(np.where(np.arange(len(v)) == order[0], top2, top1), index=df.index)

    sq["max_log_value_same_pos"] = sq.groupby(["club_id", "position"], group_keys=False)[
        ["log_value_sq"]].apply(max_excl)
    sq["n_same_pos"] = sq["pos_n"] - 1
    sq["log_sum_value_same_pos"] = np.log1p(sq["pos_sum"] - sq["value_f"])
    sq["n_signings_same_pos"] = sq["pos_sign"] - sq["signing"]
    cols = ["player_id", "club_id", "n_same_pos", "max_log_value_same_pos", "log_sum_value_same_pos",
            "n_signings_same_pos", "value_rank_in_pos"]
    out = pop[["player_id", "club_id"]].merge(sq[cols], on=["player_id", "club_id"], how="left")
    out["log_club_squad_value"] = np.log1p(out["club_id"].map(club_val))
    return out


def _own_value(tables, pop, snap):
    vals = tables["vals"]
    n = len(pop)
    v0, d0 = asof_value(vals, pop["player_id"], np.full(n, snap))
    v6, _ = asof_value(vals, pop["player_id"], np.full(n, snap - pd.Timedelta(days=182)))
    v12, _ = asof_value(vals, pop["player_id"], np.full(n, snap - pd.Timedelta(days=365)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "player_id": pop["player_id"].to_numpy(),
            "log_value": np.log(v0),
            "value_age_days": (snap - pd.to_datetime(d0)).days,
            "value_change_6m": np.log(v0) - np.log(v6),
            "value_change_12m": np.log(v0) - np.log(v12),
        })
    return out.replace([np.inf, -np.inf], np.nan)


def build_features_one(tables, pop):
    """Features for one league-season population."""
    league, season = pop["league"].iat[0], pop["season"].iat[0]
    snap = snapshot_date(season)
    keys = ["player_id", "club_id"]
    ppt = _prior_playing_time(tables, pop, league, season, snap)
    cfs = _competition_for_spot(tables, pop, league, season, snap)
    own = _own_value(tables, pop, snap)
    out = (pop.merge(ppt[keys + [c for c in NUMERIC if c in ppt.columns]], on=keys, how="left")
              .merge(cfs, on=keys, how="left")
              .merge(own, on="player_id", how="left")
              .merge(tables["players"][["player_id", "position", "sub_position", "foot", "height_in_cm"]],
                     on="player_id", how="left"))
    return out


def build_features(tables, snapshots):
    parts = [build_features_one(tables, g) for _, g in snapshots.groupby(["league", "season"], sort=True)]
    return pd.concat(parts, ignore_index=True)
