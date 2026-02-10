# Getting the data yourself

**Use one of these, not both.**

| What you need | Command | Notes |
|---------------|--------|--------|
| **2023 + 2024 stats (for prior-year forecasting)** | `pip install nflreadpy` then `python scripts/fetch_historical_nflverse.py` | **Recommended.** Use **nflreadpy** (works on Python 3.13). nfl_data_py often fails on 3.13 due to old pandas. Writes `data/real_stats_2023.csv` and `data/real_stats_2024.csv`. Keep existing `real_players.csv` and `real_teams.csv` or refresh with data_collection.py. |
| **2024 only + players + teams** | `python scripts/data_collection.py` | Uses Sleeper API. Writes players, teams, and 2024 stats. May or may not return 2023; if it doesn't, you only get 2024. |
| **2024 + 2025 rosters (team for predictions)** | `pip install nflreadpy` then `python scripts/fetch_rosters.py` | Writes `data/rosters_2024.csv` and `data/rosters_2025.csv`. Training uses 2024 rosters; app uses 2025 rosters for team strength when projecting 2025. |

**Summary:** For "prior year + situation → this year" forecasting, run **`fetch_historical_nflverse.py`** so you have 2023 and 2024. Then run **`fetch_rosters.py`** for 2024/2025 team context. Use **`data_collection.py`** if you only need 2024 and want to refresh players/teams from Sleeper.
