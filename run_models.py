"""Step 2: train and evaluate the models with rolling-origin validation.

Run:
  python run_models.py --train dk           # train on Superliga only
  python run_models.py --train all          # train on Superliga + comparable leagues (D10)
  python run_models.py --train dk --final   # ONLY when model and threshold are locked: test on 2024-2025
"""
import argparse

import pandas as pd

from src import config
from src.evaluate import plots, rolling_predictions, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", choices=["dk", "all"], default="dk")
    ap.add_argument("--final", action="store_true", help="evaluate on the hold-out seasons")
    args = ap.parse_args()

    leagues = [config.TARGET_LEAGUE] if args.train == "dk" else config.ALL_LEAGUES
    seasons = config.HOLDOUT_TEST_SEASONS if args.final else config.SELECTION_TEST_SEASONS
    tag = f"{args.train}_{'holdout' if args.final else 'selection'}"
    out = config.OUT_DIR / tag
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.OUT_DIR / "dataset.csv")
    df = df[df["season"] >= config.FIRST_SNAPSHOT]
    print(f"Training on {leagues}, testing on Danish seasons {seasons}")

    preds, params, imp = rolling_predictions(df, leagues, seasons)
    overall, per_season = summarise(preds)

    preds.to_csv(out / "predictions.csv", index=False)
    params.to_csv(out / "chosen_hyperparameters.csv", index=False)
    overall.to_csv(out / "metrics.csv", index=False)
    per_season.to_csv(out / "pr_auc_by_season.csv")
    if not imp.empty:
        (imp.groupby(["model", "feature"])["importance"].mean().reset_index()
            .sort_values(["model", "importance"], ascending=[True, False])
            .to_csv(out / "permutation_importance.csv", index=False))
    plots(preds, out)

    pd.set_option("display.width", 200)
    print("\nPooled metrics:")
    print(overall.round(3).to_string(index=False))
    print("\nPR-AUC by test season:")
    print(per_season.to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
