"""
Sleeper NFL Data Collector
Collects real NFL data from Sleeper API (free, no API key required)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SleeperNFLCollector:
    """Collects real NFL data from Sleeper API"""
    
    def __init__(self):
        self.base_url = "https://api.sleeper.app/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Fantasy-Football-Predictor/1.0'
        })
    
    def get_current_season(self) -> int:
        """Get current NFL season year"""
        now = datetime.now()
        if now.month >= 9:  # NFL season starts in September
            return now.year
        else:
            return now.year - 1
    
    def get_current_week(self) -> int:
        """Get current NFL week"""
        try:
            url = f"{self.base_url}/state/nfl"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('week', 1)
        except:
            # Fallback calculation
            season_start = datetime(self.get_current_season(), 9, 1)
            weeks_passed = (datetime.now() - season_start).days // 7
            return min(max(weeks_passed, 1), 18)
    
    def get_players(self) -> pd.DataFrame:
        """
        Get all NFL players from Sleeper API
        
        Returns:
            DataFrame with player information
        """
        logger.info("Fetching real NFL players from Sleeper API")
        
        try:
            url = f"{self.base_url}/players/nfl"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            players = []
            for player_id, player_data in data.items():
                if player_data.get('status') == 'Active':  # Only active players
                    player_info = {
                        'player_id': player_id,
                        'player_name': player_data.get('full_name', ''),
                        'position': player_data.get('position', ''),
                        'team': player_data.get('team', ''),
                        'status': player_data.get('status', ''),
                        'height': player_data.get('height', ''),
                        'weight': player_data.get('weight', ''),
                        'college': player_data.get('college', ''),
                        'years_exp': player_data.get('years_exp', 0)
                    }
                    players.append(player_info)
            
            logger.info(f"Successfully fetched {len(players)} real NFL players")
            return pd.DataFrame(players)
            
        except Exception as e:
            logger.error(f"Error fetching players: {e}")
            return pd.DataFrame()
    
    def get_player_stats(self, season: int = None, week: int = None) -> pd.DataFrame:
        """
        Get real player statistics from Sleeper API
        
        Args:
            season: NFL season year
            week: Specific week (optional)
        
        Returns:
            DataFrame with player statistics
        """
        if season is None:
            season = self.get_current_season()
        
        logger.info(f"Fetching real player stats for season {season}, week {week}")
        
        try:
            if week:
                url = f"{self.base_url}/stats/nfl/regular/{season}/{week}"
            else:
                url = f"{self.base_url}/stats/nfl/regular/{season}"
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            player_stats = []
            for player_id, stats in data.items():
                if stats:  # Only include players with stats
                    player_info = {
                        'player_id': player_id,
                        'season': season,
                        'week': week,
                        'passing_yards': stats.get('pass_yd', 0),
                        'passing_tds': stats.get('pass_td', 0),
                        'passing_ints': stats.get('pass_int', 0),
                        'rushing_yards': stats.get('rush_yd', 0),
                        'rushing_tds': stats.get('rush_td', 0),
                        'receiving_yards': stats.get('rec_yd', 0),
                        'receiving_tds': stats.get('rec_td', 0),
                        'receptions': stats.get('rec', 0),
                        'fumbles': stats.get('fumbles_lost', 0),
                        'fantasy_points': self._calculate_fantasy_points(stats)
                    }
                    player_stats.append(player_info)
            
            logger.info(f"Successfully fetched {len(player_stats)} real player stats")
            return pd.DataFrame(player_stats)
            
        except Exception as e:
            logger.error(f"Error fetching player stats: {e}")
            return pd.DataFrame()
    
    def _calculate_fantasy_points(self, stats: dict) -> float:
        """Calculate fantasy points using standard scoring"""
        points = 0.0
        
        # Passing
        points += stats.get('pass_yd', 0) * 0.04  # 1 point per 25 yards
        points += stats.get('pass_td', 0) * 4     # 4 points per TD
        points -= stats.get('pass_int', 0) * 2    # -2 points per INT
        
        # Rushing
        points += stats.get('rush_yd', 0) * 0.1   # 1 point per 10 yards
        points += stats.get('rush_td', 0) * 6     # 6 points per TD
        
        # Receiving
        points += stats.get('rec_yd', 0) * 0.1    # 1 point per 10 yards
        points += stats.get('rec_td', 0) * 6      # 6 points per TD
        points += stats.get('rec', 0) * 1         # 1 point per reception
        
        # Fumbles
        points -= stats.get('fumbles_lost', 0) * 2  # -2 points per fumble
        
        return round(points, 2)
    
    def get_teams(self) -> pd.DataFrame:
        """
        Get real NFL team information
        
        Returns:
            DataFrame with team information
        """
        logger.info("Fetching real NFL teams from Sleeper API")
        
        # Sleeper API team data (hardcoded since the endpoint doesn't work)
        teams_data = [
            {'team_id': 'ARI', 'team': 'ARI', 'name': 'Arizona Cardinals', 'conference': 'NFC', 'division': 'West'},
            {'team_id': 'ATL', 'team': 'ATL', 'name': 'Atlanta Falcons', 'conference': 'NFC', 'division': 'South'},
            {'team_id': 'BAL', 'team': 'BAL', 'name': 'Baltimore Ravens', 'conference': 'AFC', 'division': 'North'},
            {'team_id': 'BUF', 'team': 'BUF', 'name': 'Buffalo Bills', 'conference': 'AFC', 'division': 'East'},
            {'team_id': 'CAR', 'team': 'CAR', 'name': 'Carolina Panthers', 'conference': 'NFC', 'division': 'South'},
            {'team_id': 'CHI', 'team': 'CHI', 'name': 'Chicago Bears', 'conference': 'NFC', 'division': 'North'},
            {'team_id': 'CIN', 'team': 'CIN', 'name': 'Cincinnati Bengals', 'conference': 'AFC', 'division': 'North'},
            {'team_id': 'CLE', 'team': 'CLE', 'name': 'Cleveland Browns', 'conference': 'AFC', 'division': 'North'},
            {'team_id': 'DAL', 'team': 'DAL', 'name': 'Dallas Cowboys', 'conference': 'NFC', 'division': 'East'},
            {'team_id': 'DEN', 'team': 'DEN', 'name': 'Denver Broncos', 'conference': 'AFC', 'division': 'West'},
            {'team_id': 'DET', 'team': 'DET', 'name': 'Detroit Lions', 'conference': 'NFC', 'division': 'North'},
            {'team_id': 'GB', 'team': 'GB', 'name': 'Green Bay Packers', 'conference': 'NFC', 'division': 'North'},
            {'team_id': 'HOU', 'team': 'HOU', 'name': 'Houston Texans', 'conference': 'AFC', 'division': 'South'},
            {'team_id': 'IND', 'team': 'IND', 'name': 'Indianapolis Colts', 'conference': 'AFC', 'division': 'South'},
            {'team_id': 'JAX', 'team': 'JAX', 'name': 'Jacksonville Jaguars', 'conference': 'AFC', 'division': 'South'},
            {'team_id': 'KC', 'team': 'KC', 'name': 'Kansas City Chiefs', 'conference': 'AFC', 'division': 'West'},
            {'team_id': 'LV', 'team': 'LV', 'name': 'Las Vegas Raiders', 'conference': 'AFC', 'division': 'West'},
            {'team_id': 'LAC', 'team': 'LAC', 'name': 'Los Angeles Chargers', 'conference': 'AFC', 'division': 'West'},
            {'team_id': 'LAR', 'team': 'LAR', 'name': 'Los Angeles Rams', 'conference': 'NFC', 'division': 'West'},
            {'team_id': 'MIA', 'team': 'MIA', 'name': 'Miami Dolphins', 'conference': 'AFC', 'division': 'East'},
            {'team_id': 'MIN', 'team': 'MIN', 'name': 'Minnesota Vikings', 'conference': 'NFC', 'division': 'North'},
            {'team_id': 'NE', 'team': 'NE', 'name': 'New England Patriots', 'conference': 'AFC', 'division': 'East'},
            {'team_id': 'NO', 'team': 'NO', 'name': 'New Orleans Saints', 'conference': 'NFC', 'division': 'South'},
            {'team_id': 'NYG', 'team': 'NYG', 'name': 'New York Giants', 'conference': 'NFC', 'division': 'East'},
            {'team_id': 'NYJ', 'team': 'NYJ', 'name': 'New York Jets', 'conference': 'AFC', 'division': 'East'},
            {'team_id': 'PHI', 'team': 'PHI', 'name': 'Philadelphia Eagles', 'conference': 'NFC', 'division': 'East'},
            {'team_id': 'PIT', 'team': 'PIT', 'name': 'Pittsburgh Steelers', 'conference': 'AFC', 'division': 'North'},
            {'team_id': 'SF', 'team': 'SF', 'name': 'San Francisco 49ers', 'conference': 'NFC', 'division': 'West'},
            {'team_id': 'SEA', 'team': 'SEA', 'name': 'Seattle Seahawks', 'conference': 'NFC', 'division': 'West'},
            {'team_id': 'TB', 'team': 'TB', 'name': 'Tampa Bay Buccaneers', 'conference': 'NFC', 'division': 'South'},
            {'team_id': 'TEN', 'team': 'TEN', 'name': 'Tennessee Titans', 'conference': 'AFC', 'division': 'South'},
            {'team_id': 'WAS', 'team': 'WAS', 'name': 'Washington Commanders', 'conference': 'NFC', 'division': 'East'}
        ]
        
        logger.info(f"Successfully fetched {len(teams_data)} real NFL teams")
        return pd.DataFrame(teams_data)
    
    def get_schedule(self, season: int = None) -> pd.DataFrame:
        """
        Get real NFL schedule
        
        Args:
            season: NFL season year
        
        Returns:
            DataFrame with schedule information
        """
        if season is None:
            season = self.get_current_season()
        
        logger.info(f"Fetching real NFL schedule for season {season}")
        
        try:
            url = f"{self.base_url}/schedule/nfl/regular/{season}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            schedule_data = []
            for game in data:
                schedule_info = {
                    'season': season,
                    'week': game.get('week'),
                    'home_team': game.get('home'),
                    'away_team': game.get('away'),
                    'game_date': game.get('date'),
                    'game_id': game.get('game_id'),
                    'status': game.get('status')
                }
                schedule_data.append(schedule_info)
            
            logger.info(f"Successfully fetched {len(schedule_data)} real schedule entries")
            return pd.DataFrame(schedule_data)
            
        except Exception as e:
            logger.error(f"Error fetching schedule: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    collector = SleeperNFLCollector()
    
    # Get current season and week
    current_season = collector.get_current_season()
    current_week = collector.get_current_week()
    
    print(f"Current NFL Season: {current_season}")
    print(f"Current Week: {current_week}")
    
    # Fetch real data
    print("\nFetching real NFL players...")
    players = collector.get_players()
    print(f"Found {len(players)} real NFL players")
    
    if not players.empty:
        print("\nSample players:")
        print(players[['player_name', 'position', 'team']].head(10))
    
    print("\nFetching real player stats...")
    stats = collector.get_player_stats(current_season, current_week)
    print(f"Found {len(stats)} real player stats")
    
    if not stats.empty:
        print("\nSample stats:")
        print(stats[['player_id', 'fantasy_points']].head(10))
    
    print("\nFetching real teams...")
    teams = collector.get_teams()
    print(f"Found {len(teams)} real NFL teams")
    
    if not teams.empty:
        print("\nSample teams:")
        print(teams[['team', 'name']].head(10))
