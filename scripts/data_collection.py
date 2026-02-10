"""
Data Collection Script
Collects real NFL data from Sleeper API
"""

import sys
sys.path.append('../src')

from data_collection.sleeper_collector import SleeperNFLCollector
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_real_data():
    """Collect real NFL data"""
    
    logger.info("Starting real data collection...")
    
    # Initialize collector
    collector = SleeperNFLCollector()
    
    # Get real players
    logger.info("Fetching real NFL players...")
    players = collector.get_players()
    
    # Filter for active players with teams
    active_players = players[
        (players['team'].notna()) & 
        (players['team'] != 'None') & 
        (players['player_name'].notna()) &
        (players['player_name'] != '')
    ].copy()
    
    logger.info(f"Found {len(active_players)} active NFL players")
    
    # Get real teams
    logger.info("Fetching real NFL teams...")
    teams = collector.get_teams()
    logger.info(f"Found {len(teams)} NFL teams")
    
    # Get real stats for 2024
    logger.info("Fetching real 2024 player stats...")
    stats_2024 = collector.get_player_stats(2024)
    logger.info(f"Found {len(stats_2024)} real player stats for 2024")
    
    # Merge players with stats
    if not stats_2024.empty:
        # Convert player_id to string for consistent merging
        active_players['player_id'] = active_players['player_id'].astype(str)
        stats_2024['player_id'] = stats_2024['player_id'].astype(str)
        
        # Remove duplicate player_name column from stats before merging
        if 'player_name' in stats_2024.columns:
            stats_2024 = stats_2024.drop('player_name', axis=1)
        
        # Add player names to stats
        stats_2024 = stats_2024.merge(active_players[['player_id', 'player_name']], on='player_id', how='left')
        
        # Filter stats to only include players with names
        stats_with_names = stats_2024[stats_2024['player_name'].notna()].copy()
        
        logger.info(f"Stats with player names: {len(stats_with_names)}")
        
        # Save real data
        logger.info("Saving real data...")
        active_players.to_csv('../data/real_players.csv', index=False)
        teams.to_csv('../data/real_teams.csv', index=False)
        stats_with_names.to_csv('../data/real_stats_2024.csv', index=False)
        
        logger.info("Real data collection completed!")
        
        return {
            'players': active_players,
            'teams': teams,
            'stats': stats_with_names
        }
    
    else:
        logger.warning("No real stats found")
        return None

if __name__ == "__main__":
    real_data = collect_real_data()
    
    if real_data:
        print(f"\n✅ Successfully collected real NFL data!")
        print(f"   - {len(real_data['players'])} real players")
        print(f"   - {len(real_data['teams'])} real teams") 
        print(f"   - {len(real_data['stats'])} real player stats")
        print(f"\nData saved to:")
        print(f"   - data/real_players.csv")
        print(f"   - data/real_teams.csv")
        print(f"   - data/real_stats_2024.csv")
    else:
        print("❌ Failed to collect real data")
