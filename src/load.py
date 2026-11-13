"""Load the Transfermarkt tables we need.

D9.2: from players.csv we only load stable fields. Fields describing the player today
(current club, contract, current value, caps ...) are never read, so they cannot leak.
"""
import pandas as pd

from . import config

PLAYER_COLS = ["player_id", "date_of_birth", "position", "sub_position", "foot", "height_in_cm"]


def _read(name, **kwargs):
    return pd.read_csv(config.DATA_DIR / f"{name}.csv.gz", **kwargs)


def load_tables(leagues=None):
    leagues = leagues or config.ALL_LEAGUES

    comps = _read("competitions", usecols=["competition_id", "type", "domestic_league_code"])
    cups = comps[(comps["type"] == "domestic_cup") & comps["domestic_league_code"].isin(leagues)]
    cup_of = dict(zip(cups["domestic_league_code"], cups["competition_id"]))
    national = set(comps.loc[comps["type"] == "national_team_competition", "competition_id"])

    games = _read(
        "games",
        usecols=["game_id", "competition_id", "season", "date", "home_club_id", "away_club_id",
                 "home_club_goals", "away_club_goals"],
        parse_dates=["date"],
    )
    games = games[games["competition_id"].isin(set(leagues) | set(cup_of.values()))].copy()

    lineups = _read("game_lineups", usecols=["game_id", "player_id", "club_id", "type"])
    lineups = (lineups[lineups["game_id"].isin(games["game_id"])]
               .drop_duplicates(["game_id", "player_id"]))

    apps = _read(
        "appearances",
        usecols=["game_id", "player_id", "player_club_id", "date", "competition_id",
                 "goals", "assists", "yellow_cards", "red_cards", "minutes_played"],
        parse_dates=["date"],
    )
    apps = apps[~apps["competition_id"].isin(national)].copy()   # club football only

    vals = _read("player_valuations", usecols=["player_id", "date", "market_value_in_eur"],
                 parse_dates=["date"])
    vals = vals.dropna(subset=["market_value_in_eur"]).sort_values("date")

    players = _read("players", usecols=PLAYER_COLS, parse_dates=["date_of_birth"])
    players["position"] = players["position"].replace("Missing", pd.NA)

    transfers = _read(
        "transfers",
        usecols=["player_id", "transfer_date", "from_club_id", "to_club_id",
                 "from_club_name", "to_club_name"],
        parse_dates=["transfer_date"],
    )
    # Data validation: the file contains placeholder dates in the future (e.g. 2030-06-30).
    data_end = games["date"].max()
    transfers = transfers[transfers["transfer_date"] <= data_end].copy()

    return {"games": games, "lineups": lineups, "apps": apps, "vals": vals,
            "players": players, "transfers": transfers, "cup_of": cup_of}


def truncate(tables, cutoff):
    """Copy of the tables with every dated row strictly before `cutoff`. Used by the leakage test."""
    t = dict(tables)
    t["games"] = tables["games"][tables["games"]["date"] < cutoff]
    t["lineups"] = tables["lineups"][tables["lineups"]["game_id"].isin(t["games"]["game_id"])]
    t["apps"] = tables["apps"][tables["apps"]["date"] < cutoff]
    t["vals"] = tables["vals"][tables["vals"]["date"] < cutoff]
    t["transfers"] = tables["transfers"][tables["transfers"]["transfer_date"] < cutoff]
    return t
