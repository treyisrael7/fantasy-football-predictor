"""
Feature Engineering for Fantasy Football Predictions
Creates meaningful features from raw data for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Creates and engineers features for fantasy football predictions"""
    
    def __init__(self):
        self.feature_columns = []
        self.scaler = None
    
    def create_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create player-specific features
        
        Args:
            df: DataFrame with player statistics
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating player features")
        
        df = df.copy()
        
        # Rolling averages (last 3, 5 games)
        df = self._add_rolling_averages(df)
        
        # Performance trends
        df = self._add_performance_trends(df)
        
        # Efficiency metrics
        df = self._add_efficiency_metrics(df)
        
        # Opportunity metrics
        df = self._add_opportunity_metrics(df)
        
        # Consistency metrics
        df = self._add_consistency_metrics(df)
        
        return df
    
    def create_matchup_features(self, player_df: pd.DataFrame, 
                              team_df: pd.DataFrame, 
                              schedule_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create matchup-specific features
        
        Args:
            player_df: Player statistics DataFrame
            team_df: Team statistics DataFrame
            schedule_df: Schedule DataFrame
        
        Returns:
            DataFrame with matchup features
        """
        logger.info("Creating matchup features")
        
        df = player_df.copy()
        
        # Merge with schedule to get opponents (if schedule data exists)
        if not schedule_df.empty and 'team' in schedule_df.columns:
            df = df.merge(schedule_df, on=['team', 'week', 'season'], how='left')
        else:
            # Add dummy opponent data if no schedule
            df['opponent'] = 'UNK'
            df['home_team'] = df['team']
            df['away_team'] = 'UNK'
        
        # Add opponent defensive stats
        df = self._add_opponent_defense(df, team_df)
        
        # Add home/away features
        df = self._add_home_away_features(df)
        
        # Add weather features
        df = self._add_weather_features(df)
        
        return df
    
    def create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create contextual features (time-based, situational)
        
        Args:
            df: DataFrame with player and matchup data
        
        Returns:
            DataFrame with contextual features
        """
        logger.info("Creating contextual features")
        
        df = df.copy()
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Situational features
        df = self._add_situational_features(df)
        
        # Injury features
        df = self._add_injury_features(df)
        
        return df
    
    def _add_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average features"""
        df = df.sort_values(['player_name', 'week'])
        
        # Last 3 games rolling averages
        rolling_cols = ['fantasy_points', 'rushing_yards', 'receiving_yards', 'passing_yards']
        
        for col in rolling_cols:
            if col in df.columns:
                df[f'{col}_last_3_avg'] = df.groupby('player_name')[col].rolling(3, min_periods=1).mean().values
                df[f'{col}_last_5_avg'] = df.groupby('player_name')[col].rolling(5, min_periods=1).mean().values
        
        return df
    
    def _add_performance_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance trend features"""
        df = df.sort_values(['player_name', 'week'])
        
        # Calculate trend (slope of last 3 games)
        if 'fantasy_points' in df.columns:
            df['fantasy_trend'] = df.groupby('player_name')['fantasy_points'].transform(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
        
        # Performance vs season average
        df['fantasy_vs_season_avg'] = df['fantasy_points'] - df.groupby('player_name')['fantasy_points'].transform('mean')
        
        return df
    
    def _add_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add efficiency metrics"""
        
        # Yards per carry
        if 'rushing_yards' in df.columns and 'carries' in df.columns:
            df['yards_per_carry'] = df['rushing_yards'] / df['carries'].replace(0, 1)
        
        # Yards per reception
        if 'receiving_yards' in df.columns and 'receptions' in df.columns:
            df['yards_per_reception'] = df['receiving_yards'] / df['receptions'].replace(0, 1)
        
        # Completion percentage
        if 'completions' in df.columns and 'attempts' in df.columns:
            df['completion_pct'] = df['completions'] / df['attempts'].replace(0, 1)
        
        # Target share (for WR/TE)
        if 'targets' in df.columns:
            df['target_share'] = df.groupby(['team', 'week'])['targets'].transform(
                lambda x: x / x.sum() if x.sum() > 0 else 0
            )
        
        return df
    
    def _add_opportunity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add opportunity-based metrics"""
        
        # Touch share
        if 'carries' in df.columns and 'receptions' in df.columns:
            df['touches'] = df['carries'] + df['receptions']
            df['touch_share'] = df.groupby(['team', 'week'])['touches'].transform(
                lambda x: x / x.sum() if x.sum() > 0 else 0
            )
        
        # Red zone touches
        df['red_zone_touches'] = np.random.randint(0, 5, len(df))  # Placeholder
        
        return df
    
    def _add_consistency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add consistency metrics"""
        df = df.sort_values(['player_name', 'week'])
        
        # Coefficient of variation (consistency)
        if 'fantasy_points' in df.columns:
            df['fantasy_consistency'] = df.groupby('player_name')['fantasy_points'].transform(
                lambda x: x.std() / x.mean() if x.mean() > 0 else 0
            )
        
        # Boom/bust ratio
        if 'fantasy_points' in df.columns:
            season_avg = df.groupby('player_name')['fantasy_points'].transform('mean')
            df['boom_games'] = (df['fantasy_points'] > season_avg * 1.5).astype(int)
            df['bust_games'] = (df['fantasy_points'] < season_avg * 0.5).astype(int)
        
        return df
    
    def _add_opponent_defense(self, df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
        """Add opponent defensive statistics"""
        
        # Merge with team data for opponent defense
        team_df_renamed = team_df.rename(columns={
            'team': 'opponent',
            'defensive_rank': 'opp_def_rank',
            'points_against': 'opp_points_against'
        })
        
        df = df.merge(team_df_renamed[['opponent', 'opp_def_rank', 'opp_points_against']], 
                     on='opponent', how='left')
        
        # Create defensive strength features
        df['opp_def_strength'] = 33 - df['opp_def_rank']  # Higher = better defense
        df['opp_points_allowed_avg'] = df['opp_points_against'] / 16  # Per game average
        
        return df
    
    def _add_home_away_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add home/away features"""
        
        # Home field advantage
        df['is_home'] = (df['home_team'] == df['team']).astype(int)
        
        # Home/away performance splits
        df = df.sort_values(['player_name', 'week'])
        
        if 'fantasy_points' in df.columns:
            # Home performance
            home_perf = df[df['is_home'] == 1].groupby('player_name')['fantasy_points'].mean()
            df['home_avg'] = df['player_name'].map(home_perf).fillna(df['fantasy_points'].mean())
            
            # Away performance
            away_perf = df[df['is_home'] == 0].groupby('player_name')['fantasy_points'].mean()
            df['away_avg'] = df['player_name'].map(away_perf).fillna(df['fantasy_points'].mean())
        
        return df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather-related features"""
        
        # Simulate weather data
        np.random.seed(42)
        df['temperature'] = np.random.randint(20, 80, len(df))
        df['wind_speed'] = np.random.randint(0, 25, len(df))
        df['precipitation'] = np.random.choice(['None', 'Light Rain', 'Heavy Rain', 'Snow'], len(df))
        df['is_dome'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
        
        # Weather impact features
        df['cold_weather'] = (df['temperature'] < 40).astype(int)
        df['windy_conditions'] = (df['wind_speed'] > 15).astype(int)
        df['bad_weather'] = (df['precipitation'] != 'None').astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        # Week of season
        df['week_of_season'] = df['week']
        
        # Late season games (playoff push)
        df['late_season'] = (df['week'] >= 14).astype(int)
        
        # Bye week impact (games after bye)
        df['weeks_since_bye'] = np.random.randint(0, 8, len(df))  # Placeholder
        
        return df
    
    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational features"""
        
        # Game script (estimated based on team records)
        df['favorite'] = np.random.choice([0, 1], len(df))  # Placeholder
        df['underdog'] = 1 - df['favorite']
        
        # Divisional games
        df['divisional_game'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        
        # Prime time games
        df['prime_time'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
        
        return df
    
    def _add_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add injury-related features"""
        
        # Injury status
        df['injury_status'] = np.random.choice(['Healthy', 'Questionable', 'Doubtful', 'Out'], 
                                             len(df), p=[0.7, 0.15, 0.1, 0.05])
        
        # Injury impact
        injury_impact = {
            'Healthy': 1.0,
            'Questionable': 0.8,
            'Doubtful': 0.5,
            'Out': 0.0
        }
        df['injury_multiplier'] = df['injury_status'].map(injury_impact)
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, target_type: str = 'next_game') -> pd.DataFrame:
        """
        Create target variable for prediction
        
        Args:
            df: DataFrame with player data
            target_type: Type of target ('next_game', 'rest_of_season')
        
        Returns:
            DataFrame with target variable
        """
        logger.info(f"Creating target variable: {target_type}")
        
        df = df.copy()
        df = df.sort_values(['player_name', 'week'])
        
        if target_type == 'next_game':
            # Next game fantasy points
            df['target_fantasy_points'] = df.groupby('player_name')['fantasy_points'].shift(-1)
        elif target_type == 'rest_of_season':
            # Average fantasy points for rest of season
            df['target_fantasy_points'] = df.groupby('player_name')['fantasy_points'].apply(
                lambda x: x.iloc[1:].mean() if len(x) > 1 else x.iloc[0]
            ).values
        
        # Remove rows where target is NaN
        df = df.dropna(subset=['target_fantasy_points'])
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of engineered feature columns"""
        return [
            'fantasy_points_last_3_avg', 'fantasy_points_last_5_avg',
            'fantasy_trend', 'fantasy_vs_season_avg',
            'yards_per_carry', 'yards_per_reception', 'completion_pct',
            'target_share', 'touch_share', 'red_zone_touches',
            'fantasy_consistency', 'boom_games', 'bust_games',
            'opp_def_strength', 'opp_points_allowed_avg',
            'is_home', 'home_avg', 'away_avg',
            'cold_weather', 'windy_conditions', 'bad_weather', 'is_dome',
            'week_of_season', 'late_season', 'weeks_since_bye',
            'favorite', 'underdog', 'divisional_game', 'prime_time',
            'injury_multiplier'
        ]

# Example usage
if __name__ == "__main__":
    from data_collection.sleeper_collector import SleeperNFLCollector
    
    collector = SleeperNFLCollector()
    player_df = collector.get_player_stats(2024, 1)
    team_df = collector.get_teams()
    schedule_df = collector.get_schedule(2024)
    
    # Engineer features
    engineer = FeatureEngineer()
    
    # Create features
    player_df = engineer.create_player_features(player_df)
    player_df = engineer.create_matchup_features(player_df, team_df, schedule_df)
    player_df = engineer.create_contextual_features(player_df)
    player_df = engineer.create_target_variable(player_df)
    
    print("Engineered features:")
    print(player_df.columns.tolist())
    print(f"\nDataset shape: {player_df.shape}")
    print(f"Features created: {len(engineer.get_feature_columns())}")
