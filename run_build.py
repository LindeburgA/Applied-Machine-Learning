"""Step 1: build the dataset (population + target + features) and save it to output/dataset.csv.

Run:  python run_build.py
"""
import time

import pandas as pd

from src import config
from src.features import build_features
from src.load import load_tables
from src.snapshot import build_snapshots


def main():
    t0 = time.time()
    config.OUT_DIR.mkdir(exist_ok=True)
    print("Loading tables ...")
    tables = load_tables()
    print(f"  done in {time.time() - t0:.0f}s")

    print("Building population and target ...")
    snaps = build_snapshots(tables)
    print("Building features ...")
    data = build_features(tables, snaps)
    data.to_csv(config.OUT_DIR / "dataset.csv", index=False)

    # Data validation summary (goes into the report, D4 and D7)
    status = pd.crosstab([data["league"], data["season"]], data["left_club"])
    status.to_csv(config.OUT_DIR / "population_by_season.csv")
    stayed = data[data["left_club"] == "stayed"]
    rate = stayed.groupby(["league", "season"])["target"].agg(n="size", positive_rate="mean")
    rate.to_csv(config.OUT_DIR / "target_rate_by_season.csv")

    dk = data[data["league"] == config.TARGET_LEAGUE]
    print(f"\nRows: {len(data)}  |  Danish rows: {len(dk)}")
    print("\nDanish population by leaving status (D7):")
    print(dk["left_club"].value_counts().to_string())
    print("\nDanish positive rate among players who stayed:")
    print(rate.loc[config.TARGET_LEAGUE].round(2).to_string())
    print(f"\nSaved to {config.OUT_DIR}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
