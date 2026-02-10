"""
Data Collection Script
Collects real NFL data from Sleeper API (and optionally 2023 for prior-year forecasting).
"""

import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from data_collection.sleeper_collector import SleeperNFLCollector
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(_root, "data")


def collect_real_data(seasons=None):
    """Collect real NFL data from Sleeper. seasons: list of years, e.g. [2023, 2024]."""
    if seasons is None:
        seasons = [2024]

    logger.info("Starting real data collection...")
    collector = SleeperNFLCollector()

    logger.info("Fetching real NFL players...")
    players = collector.get_players()
    active_players = players[
        (players['team'].notna()) &
        (players['team'] != 'None') &
        (players['player_name'].notna()) &
        (players['player_name'] != '')
    ].copy()
    logger.info(f"Found {len(active_players)} active NFL players")

    logger.info("Fetching real NFL teams...")
    teams = collector.get_teams()
    logger.info(f"Found {len(teams)} NFL teams")

    active_players['player_id'] = active_players['player_id'].astype(str)
    os.makedirs(DATA_DIR, exist_ok=True)
    active_players.to_csv(os.path.join(DATA_DIR, "real_players.csv"), index=False)
    teams.to_csv(os.path.join(DATA_DIR, "real_teams.csv"), index=False)

    result = {'players': active_players, 'teams': teams, 'stats_by_season': {}}

    for season in seasons:
        logger.info("Fetching real %s player stats...", season)
        stats_df = collector.get_player_stats(season)
        logger.info("Found %s real player stats for %s", len(stats_df), season)
        if stats_df.empty:
            continue
        stats_df['player_id'] = stats_df['player_id'].astype(str)
        if 'player_name' in stats_df.columns:
            stats_df = stats_df.drop('player_name', axis=1)
        stats_df = stats_df.merge(
            active_players[['player_id', 'player_name']], on='player_id', how='left'
        )
        stats_with_names = stats_df[stats_df['player_name'].notna()].copy()
        path = os.path.join(DATA_DIR, f"real_stats_{season}.csv")
        stats_with_names.to_csv(path, index=False)
        result['stats_by_season'][season] = stats_with_names
        logger.info("Saved %s rows to %s", len(stats_with_names), path)

    logger.info("Real data collection completed!")
    return result


if __name__ == "__main__":
    # Fetch 2023, 2024 (for training), and 2025 (for actuals / evaluation)
    data = collect_real_data(seasons=[2023, 2024, 2025])
    n_players = len(data["players"])
    n_teams = len(data["teams"])
    by_season = data["stats_by_season"]
    if by_season:
        print("\n✅ Successfully collected real NFL data!")
        print(f"   - {n_players} players, {n_teams} teams")
        for year, df in by_season.items():
            print(f"   - {len(df)} stats for {year} → data/real_stats_{year}.csv")
        if 2023 not in by_season and 2024 in by_season:
            print("\n💡 No 2023 data from Sleeper. For prior-year data, run: python scripts/fetch_historical_nflverse.py")
        if 2025 in by_season:
            print("   - 2025 actuals from Sleeper will be used for evaluation in the app.")
    else:
        print("❌ No stats returned. Try scripts/fetch_historical_nflverse.py for 2023/2024 data.")
