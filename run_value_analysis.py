"""Step 3 (sub-question 4): do young players with a lot of playing time rise more in value?

Descriptive only. Value is the business argument, not the target, and this uses values AFTER
the snapshot on purpose - it is never used as a model feature. It shows association, not causation.

Run:  python run_value_analysis.py
"""
import numpy as np
import pandas as pd

from src import config
from src.features import asof_value
from src.load import load_tables


def main():
    df = pd.read_csv(config.OUT_DIR / "dataset.csv", parse_dates=["snapshot_date"])
    df = df[(df["league"] == config.TARGET_LEAGUE) & (df["left_club"] == "stayed")
            & (df["season"] >= config.FIRST_SNAPSHOT)].copy()
    vals = load_tables([config.TARGET_LEAGUE])["vals"]

    later = (df["snapshot_date"] + pd.Timedelta(days=365)).to_numpy()
    v_end, _ = asof_value(vals, df["player_id"], later)
    df["log_value_change_12m_after"] = np.log(v_end) - df["log_value"]
    df = df.dropna(subset=["log_value_change_12m_after"])
    df["pct_change"] = np.exp(df["log_value_change_12m_after"]) - 1

    df["minutes_band"] = pd.cut(df["share_next"], [-0.01, 0.10, 0.30, 0.50, 1.0],
                                labels=["0-10%", "10-30%", "30-50%", "50-100%"])
    by_band = df.groupby("minutes_band", observed=True)["pct_change"].agg(
        n="size", median="median", mean="mean",
        share_rising=lambda x: (x > 0).mean()).round(3)
    by_target = df.groupby("target")["pct_change"].agg(n="size", median="median", mean="mean").round(3)

    by_band.to_csv(config.OUT_DIR / "value_change_by_minutes.csv")
    by_target.to_csv(config.OUT_DIR / "value_change_by_target.csv")
    print("Value change over the 12 months after 1 July, Danish players who stayed:\n")
    print(by_band.to_string())
    print("\nBy target (1 = reached 30% of league minutes):\n")
    print(by_target.to_string())


if __name__ == "__main__":
    main()
