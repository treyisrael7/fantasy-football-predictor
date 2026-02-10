# Model improvements: what we use and what would help

## What we use now

- **Prior-year PPG** – strongest signal; model weights it heavily.
- **Stat totals** – passing/rushing/receiving yards and TDs, receptions, interceptions, fumbles.
- **Team strength** – average PPG of teammates (same team in target year when we have rosters).
- **Position** – encoded (QB, RB, WR, TE, K, DEF).
- **Years of experience** – from `real_players.csv` (`years_exp`); helps with age/experience curve.
- **Sample weighting** – training uses all players but weights by target PPG so fantasy-relevant players (higher PPG) count more in the loss.
- **Optional stats from nflverse** – if you re-run `fetch_historical_nflverse.py`, the script now tries to save `targets`, `carries`, `games_played`, and `passing_attempts` when the API returns them. Adding these as features later would be a small code change (add to `FEATURE_COLUMNS` and to the merge/aggregation in training and the app).

## What would help if we had the data

- **Volume / role**
  - **Targets** (WR/TE/RB) – already supported in fetch; add to features when present.
  - **Carries** – same.
  - **Snap share or participation** – not in basic nflverse player stats; would need another source (e.g. nflverse snap counts or PFF if you have access).

- **Durability**
  - **Games played / games missed** – `games_played` is saved when we have weekly data; can add as a feature (e.g. `games_missed = 17 - games_played`).
  - **Injury history** – would require injury reports or games-missed by season from a separate dataset.

- **Context**
  - **Strength of schedule** – defensive rankings or Vegas lines by team; would need to pull and join by opponent.
  - **Coaching / OC change** – would need a manual or scraped table (e.g. “new OC in 2024”).
  - **Team pass/rush rate, pace** – derivable from nflverse play-by-play or drive-level data if we load it.

- **Other**
  - **Vegas lines** (team totals, win totals) – good proxy for team quality; would need an external API or scrape.
  - **Ranking loss** – train or evaluate on “did we rank the better player higher?” (e.g. pairwise or Spearman) instead of only MSE/MAE; helps for draft order.

## Quick wins with current data

1. **Re-run fetch** – `python scripts/fetch_historical_nflverse.py` so new stats CSVs include `targets`, `carries`, `games_played`, `passing_attempts` when available.
2. **Add those columns to the model** – add to `FEATURE_COLUMNS` in `train_models.py` and to the app’s feature list; in training, use `*_prev` for prior-year targets/carries/games_played/attempts.
3. **Add `games_missed`** – if `games_played` exists, add `games_missed = 17 - games_played` (or 16/18 depending on season length) as a durability feature.
