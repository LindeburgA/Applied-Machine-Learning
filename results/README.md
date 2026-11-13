# Resultater fra den oprindelige modelkørsel

Denne mappe deler projektets **samlede tabeller og figurer** med projektgruppen. Filerne er kopieret fra den eksisterende kørsel i `playing_time/output/`. De er ikke nye modelkørsler.

## Hvad der blev målt

For Superligaen indeholder den anvendte dataversion 22 ligakampe pr. klub i 2024/25 og 2025/26. Målet i disse resultater er derfor mindst **30 % af de registrerede grundspilsminutter**. Slutspillets runde 23–32 mangler. Fortolk ikke tallene som hele sæsonens spilletid.

Spillere, der ifølge transfer- eller kampdata forlod klubben, blev udeladt fra modeltræning og evaluering. Den tilbageværende gruppe er derfor ikke nødvendigvis repræsentativ for alle unge spillere, klubben skulle beslutte om.

## Hovedtal

| Dansk test | Antal spillere | Gennembrud i datasættet | Logistisk regression PR-AUC | Tommelfingerregel PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Udvælgelse 2020–2023 | 323 | 100 | 0,754 | 0,743 |
| Tidligere sluttest 2024–2025 | 142 | 48 | 0,805 | 0,718 |

I den tidligere sluttest og ved beslutningsgrænsen 2:1 anbefalede den logistiske model at beholde 33 spillere: 27 nåede målet, 6 gjorde ikke. Tommelfingerreglen anbefalede at beholde 22: 18 nåede målet, 4 gjorde ikke. Modellen fandt 9 flere korrekte behold, men gav også 2 flere forkerte behold. Begge regler havde 81,82 % precision. Modellens recall var 56,25 % mod 37,50 % for tommelfingerreglen.

Beslutningsgrænsen bygger på **en antagelse** om, at et forkert behold koster dobbelt så meget som et forkert lån. De beregnede fejlomkostninger er ikke kroner. Sluttesten er allerede set og kan ikke bruges som en ny, urørt test efter ændringer i model eller grænse.

## Find filerne

- [`dk_selection/`](dk_selection/) viser dansk træning og test på 2020–2023.
- [`all_selection/`](all_selection/) viser træning på de fem ligaer og test på Danmark i 2020–2023.
- [`dk_holdout/`](dk_holdout/) viser den tidligere sluttest på 2024–2025.
- [`population_by_season.csv`](population_by_season.csv) og [`target_rate_by_season.csv`](target_rate_by_season.csv) beskriver population og målrate.
- [`value_change_by_minutes.csv`](value_change_by_minutes.csv) og [`value_change_by_target.csv`](value_change_by_target.csv) er beskrivende analyser af sammenhængen mellem minutter og værdi. De viser ikke, at spilletid *forårsager* værdistigning.

Hver modelmappe indeholder `metrics.csv`, `pr_auc_by_season.csv`, `chosen_hyperparameters.csv`, `permutation_importance.csv`, `pr_curves.png` og `calibration.png`.

Rådata, den store række-for-række-modeltabel og individuelle spillerforudsigelser er ikke lagt i dette offentlige repository. De findes fortsat lokalt i det oprindelige projekts `data/` og `output/`.
