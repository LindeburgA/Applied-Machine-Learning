# Spilletid for unge Superliga-spillere

Python-projekt til at forudsige, om en spiller på 17–21 år får mindst 30 % af sin klubs **registrerede ligaminutter i grundspillet** i den kommende sæson. Hver forudsigelse bruger oplysninger, som var tilgængelige før 1. juli. Projektet bygger på beslutningerne i den tilhørende Decision Log, men rådata dækker ikke Superligaens slutspil, så målvariablen er smallere end den oprindelige formulering om hele sæsonen.

## Vigtig dataafgrænsning

Den anvendte version af Transfermarkt-datasættet indeholder 22 Superliga-kampe pr. klub i 2024/25 og 2025/26. Slutspillets runde 23–32 og spillerminutterne fra disse kampe findes ikke i `games.csv.gz` og `appearances.csv.gz`. Brug derfor ikke projektets tal som resultater for hele sæsonen. `audit_results.py` kontrollerer dækningen og læser gemte forudsigelser, hvis de findes.

2024–2025 blev allerede brugt som sluttest i det oprindelige projekt. Nye valg af features, model eller beslutningsgrænse kan undersøges på de tidligere udvælgelsessæsoner, men de to år kan ikke kaldes en ny, urørt sluttest.

## Kom i gang i VS Code

1. Åbn denne mappe i VS Code med **File → Open Folder**.
2. Åbn **Terminal → New Terminal**.
3. Opret og aktivér et Python-miljø på Mac:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

4. Vælg om nødvendigt `.venv/bin/python` via **Cmd+Shift+P → Python: Select Interpreter**.
5. Læg de komprimerede Transfermarkt-filer i `data/`: `appearances.csv.gz`, `competitions.csv.gz`, `game_lineups.csv.gz`, `games.csv.gz`, `player_valuations.csv.gz`, `players.csv.gz` og `transfers.csv.gz`. Projektet bygger på [Football Data from Transfermarkt](https://github.com/dcaribou/transfermarkt-datasets). Rådata er ikke med på GitHub.

Hvis rådata allerede ligger i en anden mappe, kan du pege på den uden at kopiere filerne:

```bash
export PLAYING_TIME_DATA_DIR="/fuld/sti/til/playing_time/data"
```

## Kør projektet

```bash
python -m pytest tests -v
python run_build.py
python run_models.py --train dk
python run_value_analysis.py
python audit_results.py
```

`run_build.py` laver `output/dataset.csv`. `run_models.py --train dk` tester på 2020–2023 med tidsopdeling. `run_value_analysis.py` er en beskrivende analyse af værdi og spilletid. `audit_results.py` kontrollerer antal kampe og viser beslutningstallene, hvis `output/dk_holdout/predictions.csv` findes.

Den oprindelige kode har også `--train all` til en sammenligning med andre ligaer og `--final` til 2024–2025. Da sluttesten allerede er brugt, skal et nyt forsøg ikke præsenteres som en urørt test. Denne version understøtter **ikke** `--features extended`.

## Filer

| Fil | Formål |
| --- | --- |
| `src/config.py` | Beslutningsgrænser, sæsoner og stier |
| `src/load.py` | Indlæsning af CSV-filer |
| `src/snapshot.py` | Udvælgelse af spillere og målvariabel |
| `src/features.py` | Oplysninger kendt før 1. juli |
| `src/models.py` | De tre modeller |
| `src/evaluate.py` | Tidsopdelt evaluering og figurer |
| `tests/` | Kontrol af datalækage |
| `audit_results.py` | Kontrol af datadækning og gemte beslutninger |

`data/`, `output/` og `.venv/` holdes uden for GitHub.
