"""Build the snapshot table: who is in the population (D4) and what happened next (D5, D7)."""
import numpy as np
import pandas as pd

from . import config


def snapshot_date(season):
    return pd.Timestamp(year=season, month=config.SNAPSHOT_MONTH, day=config.SNAPSHOT_DAY)


def league_games(tables, league, season, before=None):
    g = tables["games"]
    g = g[(g["competition_id"] == league) & (g["season"] == season)]
    return g[g["date"] < before] if before is not None else g


def club_game_rows(games):
    """One row per club per game, with points."""
    home = games.assign(club_id=games["home_club_id"], gf=games["home_club_goals"],
                        ga=games["away_club_goals"])
    away = games.assign(club_id=games["away_club_id"], gf=games["away_club_goals"],
                        ga=games["home_club_goals"])
    out = pd.concat([home, away])[["game_id", "competition_id", "season", "date", "club_id", "gf", "ga"]]
    out["points"] = np.select([out["gf"] > out["ga"], out["gf"] == out["ga"]], [3, 1], 0)
    return out


def last_club_assignment(tables, league, season):
    """Each player's last club in league or cup matchday squads of season s-1 (D4a, D4b).

    Only clubs that played in the league in s-1 count. Returns player_id, club_id,
    last_squad_date, league_squads, cup_squads.
    """
    snap = snapshot_date(season)
    cup = tables["cup_of"].get(league)
    g = tables["games"]
    prev = g[(g["season"] == season - 1) & g["competition_id"].isin([league, cup]) & (g["date"] < snap)]
    league_clubs = set(club_game_rows(prev[prev["competition_id"] == league])["club_id"])

    lu = tables["lineups"].merge(prev[["game_id", "competition_id", "date"]], on="game_id")
    lu = lu[lu["club_id"].isin(league_clubs)].sort_values(["date", "game_id"])
    if lu.empty:
        return pd.DataFrame(columns=["player_id", "club_id", "last_squad_date", "league_squads", "cup_squads"])

    last = (lu.groupby("player_id").tail(1)[["player_id", "club_id", "date"]]
            .rename(columns={"date": "last_squad_date"}))
    at_club = lu.merge(last[["player_id", "club_id"]], on=["player_id", "club_id"])
    counts = (at_club.assign(is_league=at_club["competition_id"] == league,
                             is_cup=at_club["competition_id"] == cup)
              .groupby("player_id")[["is_league", "is_cup"]].sum()
              .rename(columns={"is_league": "league_squads", "is_cup": "cup_squads"})
              .reset_index())
    return last.merge(counts, on="player_id")


def academy_promotions(tables, season, clubs, exclude_ids):
    """Players moved from a youth or reserve team to one of `clubs` in the year before the snapshot (D4e)."""
    snap = snapshot_date(season)
    t = tables["transfers"]
    t = t[(t["to_club_id"].isin(clubs))
          & (t["transfer_date"] >= snap - pd.Timedelta(days=365)) & (t["transfer_date"] < snap)
          & t["from_club_name"].str.contains(config.YOUTH_PATTERN, regex=True, na=False)
          & ~t["player_id"].isin(exclude_ids)]
    t = t.sort_values("transfer_date").groupby("player_id").tail(1)
    return pd.DataFrame({"player_id": t["player_id"], "club_id": t["to_club_id"],
                         "last_squad_date": t["transfer_date"], "league_squads": 0, "cup_squads": 0})


def build_population(tables, league, season, include_academy=True):
    """Population for one league and snapshot year (D4a-D4e)."""
    snap = snapshot_date(season)
    nxt = league_games(tables, league, season)
    prev = league_games(tables, league, season - 1, before=snap)
    if nxt.empty or prev.empty:
        return pd.DataFrame()
    clubs_next = set(club_game_rows(nxt)["club_id"])            # D4d: club stays in the league
    clubs_prev = set(club_game_rows(prev)["club_id"])

    pop = last_club_assignment(tables, league, season)
    pop = pop[pop["club_id"].isin(clubs_next)].assign(academy_promotion=0)
    if include_academy:
        acad = academy_promotions(tables, season, clubs_next & clubs_prev, set(pop["player_id"]))
        pop = pd.concat([pop, acad.assign(academy_promotion=1)], ignore_index=True)

    pop = pop.merge(tables["players"][["player_id", "date_of_birth"]], on="player_id", how="left")
    pop["age"] = (snap - pop["date_of_birth"]).dt.days / 365.25
    pop = pop[(pop["age"] >= config.AGE_MIN) & (pop["age"] < config.AGE_MAX)]   # D4c

    pop = pop.drop(columns="date_of_birth").assign(league=league, season=season, snapshot_date=snap)
    pop["cup_squad_only"] = ((pop["league_squads"] == 0) & (pop["cup_squads"] > 0)).astype(int)
    return pop.reset_index(drop=True)


def add_target(tables, pop):
    """share_next, left_club and target for one league-season population (D5, D7)."""
    if pop.empty:
        return pop
    league, season = pop["league"].iat[0], pop["season"].iat[0]
    snap = snapshot_date(season)
    nxt = league_games(tables, league, season)
    season_end = nxt["date"].max()

    n_games = club_game_rows(nxt).groupby("club_id").size().rename("club_games_next")
    a = tables["apps"]
    mins = (a[a["game_id"].isin(nxt["game_id"])]
            .groupby(["player_id", "player_club_id"])["minutes_played"].sum()
            .rename("minutes_next").reset_index().rename(columns={"player_club_id": "club_id"}))
    pop = (pop.merge(n_games, left_on="club_id", right_index=True, how="left")
              .merge(mins, on=["player_id", "club_id"], how="left"))
    pop["minutes_next"] = pop["minutes_next"].fillna(0)
    pop["share_next"] = pop["minutes_next"] / (pop["club_games_next"] * 90)

    # D7: did the player leave the club? First move away after his last squad, up to season end.
    tr = tables["transfers"].merge(pop[["player_id", "club_id", "last_squad_date"]], on="player_id")
    tr = tr[(tr["from_club_id"] == tr["club_id"]) & (tr["to_club_id"] != tr["club_id"])
            & (tr["transfer_date"] > tr["last_squad_date"]) & (tr["transfer_date"] <= season_end)]
    leave_date = tr.groupby("player_id")["transfer_date"].min()
    pop["leave_date"] = pop["player_id"].map(leave_date)
    pre_end = pd.Timestamp(year=season, month=config.PRE_SEASON_END[0], day=config.PRE_SEASON_END[1])
    pop["left_club"] = np.select(
        [pop["leave_date"] < snap, pop["leave_date"] <= pre_end, pop["leave_date"].notna()],
        ["left_before_snapshot", "left_pre_season", "left_mid_season"], "stayed")

    # Played for another club without a recorded move -> data gap
    other = a[(a["date"] >= snap) & (a["date"] <= season_end)].merge(
        pop[["player_id", "club_id"]], on="player_id")
    gap_ids = set(other.loc[other["player_club_id"] != other["club_id"], "player_id"])
    pop.loc[(pop["left_club"] == "stayed") & pop["player_id"].isin(gap_ids), "left_club"] = "data_gap"

    stayed = pop["left_club"] == "stayed"
    pop["target"] = np.where(stayed, (pop["share_next"] >= config.THRESHOLD).astype(float), np.nan)
    pop.loc[~stayed, "share_next"] = np.nan
    return pop


def build_snapshots(tables, leagues=None, seasons=None, include_academy=True):
    leagues = leagues or config.ALL_LEAGUES
    seasons = seasons or range(config.FIRST_SNAPSHOT, config.LAST_SNAPSHOT + 1)
    parts = []
    for league in leagues:
        for s in seasons:
            pop = build_population(tables, league, s, include_academy)
            if not pop.empty:
                parts.append(add_target(tables, pop))
    out = pd.concat(parts, ignore_index=True)
    out[["player_id", "club_id"]] = out[["player_id", "club_id"]].astype("int64")
    return out
