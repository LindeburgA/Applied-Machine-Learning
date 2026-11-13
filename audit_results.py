"""Kontroller datadækning og de gemte beslutninger i playing_time-projektet.

Kør fra VS Code:
    python audit_results.py

Bruger kun Pythons standardbibliotek. Ændrer ingen eksisterende filer.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from collections import Counter, defaultdict
from pathlib import Path


def csv_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def check_games(path: Path, seasons: list[int]) -> None:
    by_season: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    last_date: dict[int, str] = {}
    rounds: dict[int, set[str]] = defaultdict(set)
    for row in csv_rows(path):
        if row["competition_id"] != "DK1" or int(row["season"]) not in seasons:
            continue
        season = int(row["season"])
        by_season[season][int(row["home_club_id"])] += 1
        by_season[season][int(row["away_club_id"])] += 1
        last_date[season] = max(last_date.get(season, ""), row["date"])
        rounds[season].add(row["round"])

    print("\nDATA: Superliga-kampe i games.csv.gz")
    for season in seasons:
        counts = Counter(by_season[season].values())
        count_text = ", ".join(f"{n} klubber med {games} kampe" for games, n in sorted(counts.items()))
        print(f"{season}/{str(season + 1)[-2:]}: {count_text}; sidste kamp {last_date.get(season, 'mangler')}")
        if not counts:
            print("  ADVARSEL: sæsonen findes ikke i filen.")
        elif max(counts) < 32:
            print("  OBS: Færre end 32 kampe pr. klub; hele sæsonen er ikke dækket.")
        if not any("23." in label or "32." in label for label in rounds[season]):
            print("  Ingen registreret 23. eller 32. spillerunde.")


def check_predictions(path: Path) -> None:
    rows = list(csv_rows(path))
    if not rows:
        raise ValueError(f"Ingen forudsigelser i {path}")
    required = {"target", "season", "share_prev", "p_logistic"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Mangler kolonner i {path.name}: {', '.join(sorted(missing))}")

    y = [int(float(row["target"])) for row in rows]
    print("\nGEMT SLUTTEST: output/dk_holdout/predictions.csv")
    print(f"Spillere: {len(rows)}; nåede 30%-målet i datasættet: {sum(y)}")
    for name, col, threshold in (
        ("Tommelfingerregel", "share_prev", 0.30),
        ("Logistisk regression ved 2:1", "p_logistic", 2 / 3),
    ):
        keep = [float(row[col]) >= threshold for row in rows]
        tp = sum(k and actual == 1 for k, actual in zip(keep, y))
        fp = sum(k and actual == 0 for k, actual in zip(keep, y))
        fn = sum(not k and actual == 1 for k, actual in zip(keep, y))
        tn = sum(not k and actual == 0 for k, actual in zip(keep, y))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        cost = 2 * fp + fn
        print(f"\n{name}")
        print(f"  Anbefalet behold: {tp + fp}; korrekt behold: {tp}; forkert behold: {fp}")
        print(f"  Forkert lån: {fn}; korrekt lån: {tn}")
        print(f"  Precision: {precision:.2%}; recall: {recall:.2%}; tænkt fejlomkostning: {cost}")

    season_counts = defaultdict(lambda: [0, 0])
    for row, actual in zip(rows, y):
        count = season_counts[int(row["season"])]
        count[0] += 1
        count[1] += actual
    print("\nMålrate pr. sæson:")
    for season, (total, positive) in sorted(season_counts.items()):
        print(f"  {season}/{str(season + 1)[-2:]}: {positive}/{total} = {positive / total:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    games = Path(os.environ.get("PLAYING_TIME_DATA_DIR", args.project / "data")) / "games.csv.gz"
    predictions = args.project / "output" / "dk_holdout" / "predictions.csv"
    if not games.is_file():
        parser.error(f"Filen findes ikke: {games}")
    check_games(games, [2024, 2025])
    if predictions.is_file():
        check_predictions(predictions)
    else:
        print(f"\nIngen gemte sluttestforudsigelser i {predictions} endnu.")


if __name__ == "__main__":
    main()
