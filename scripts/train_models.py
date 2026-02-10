"""
Model Training Script
Train ML models using real NFL data
"""

import sys
import os

# Add project src so we import from src/data_collection, not scripts/data_collection.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import joblib

from models.fantasy_predictor import FantasyPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'position', 'team_strength', 'prior_ppg', 'years_exp',
    'passing_yards', 'passing_tds', 'passing_ints', 'rushing_yards',
    'rushing_tds', 'receiving_yards', 'receiving_tds', 'receptions', 'fumbles',
    'targets', 'carries',
]

# Sample weighting: weight = 1 + clip(target, 0, 10) so higher-PPG players count more (all data used)
SAMPLE_WEIGHT_MAX = 10.0


def _build_features_and_target(player_data):
    """Build feature matrix and target. player_data must have fantasy_points, team, position, stat cols."""
    position_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'K': 5, 'DEF': 6}
    player_data = player_data.copy()
    player_data['position_encoded'] = player_data['position'].map(position_map).fillna(0).astype(int)
    player_data['position'] = player_data['position_encoded']
    player_data['prior_ppg'] = 0  # same-season mode: no prior year

    team_totals = player_data.groupby('team')['fantasy_points'].agg(['sum', 'count'])
    player_data['team_fp'] = player_data['fantasy_points']
    player_data['team_sum_others'] = player_data.apply(
        lambda r: team_totals.loc[r['team'], 'sum'] - r['team_fp'] if r['team'] in team_totals.index else 0, axis=1
    )
    player_data['team_count_others'] = player_data.apply(
        lambda r: team_totals.loc[r['team'], 'count'] - 1 if r['team'] in team_totals.index else 0, axis=1
    )
    player_data['team_strength'] = np.where(
        player_data['team_count_others'] > 0,
        player_data['team_sum_others'] / player_data['team_count_others'] / 17,
        0
    )
    player_data = player_data.drop(columns=['team_fp', 'team_sum_others', 'team_count_others'], errors='ignore')
    for col in FEATURE_COLUMNS:
        if col in player_data.columns:
            player_data[col] = player_data[col].fillna(0)
    player_data['target'] = player_data['fantasy_points'] / 17
    return player_data


def train_models():
    """Train model: if 2023+2024 exist, train prior-year -> next-year (forecast 2025); else same-season 2024."""
    data_dir = os.path.join(_root, "data")
    models_dir = os.path.join(_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    players_path = os.path.join(data_dir, "real_players.csv")
    teams_path = os.path.join(data_dir, "real_teams.csv")
    stats_2024_path = os.path.join(data_dir, "real_stats_2024.csv")
    stats_2023_path = os.path.join(data_dir, "real_stats_2023.csv")

    try:
        players = pd.read_csv(players_path)
        teams = pd.read_csv(teams_path)
        stats_2024 = pd.read_csv(stats_2024_path)
    except FileNotFoundError as e:
        logger.error("Data not found: %s. Run data_collection.py or fetch_historical_nflverse.py.", e)
        return False

    players['player_id'] = players['player_id'].astype(str)
    stats_2024['player_id'] = stats_2024['player_id'].astype(str)
    # In same-season mode we merge with players (which has player_name); drop from stats to avoid duplicate cols
    if not os.path.isfile(stats_2023_path) and 'player_name' in stats_2024.columns:
        stats_2024 = stats_2024.drop(columns=['player_name'], errors='ignore')

    # Forecast mode: 2023 features -> 2024 PPG (at inference we use 2024 data to predict 2025)
    if os.path.isfile(stats_2023_path):
        stats_2023 = pd.read_csv(stats_2023_path)
        stats_2023['player_id'] = stats_2023['player_id'].astype(str)
        # Use identity from stats (nflverse uses different player_id than Sleeper — don't merge with real_players)
        stat_cols = [c for c in stats_2024.columns if c not in ('player_id', 'season', 'week', 'player_name', 'position', 'team')]
        id_cols = [c for c in ('player_id', 'player_name', 'position', 'team') if c in stats_2024.columns]
        s24 = stats_2024[id_cols + [c for c in stat_cols if c in stats_2024.columns]].copy()
        s23 = stats_2023.rename(columns={c: c + '_prev' for c in stat_cols if c in stats_2023.columns})
        s23 = s23[[c for c in s23.columns if c == 'player_id' or c.endswith('_prev')]]
        merged = s24.merge(s23, on='player_id', how='inner')
        # Fill missing name/position/team from real_players if we have a matching id (Sleeper); else use from stats or default
        player_cols = ['player_id', 'player_name', 'position', 'team']
        if 'years_exp' in players.columns:
            player_cols.append('years_exp')
        merged = merged.merge(players[player_cols].rename(columns={'player_name': '_pn', 'position': '_pos', 'team': '_team'}), on='player_id', how='left')
        if 'player_name' not in merged.columns:
            merged['player_name'] = merged.get('_pn', merged['player_id'].astype(str))
        else:
            merged['player_name'] = merged['player_name'].fillna(merged['_pn']) if '_pn' in merged.columns else merged['player_name']
        merged['position'] = merged['_pos'] if '_pos' in merged.columns else (merged['position'] if 'position' in merged.columns else 0)
        merged['team'] = merged['_team'] if '_team' in merged.columns else (merged['team'] if 'team' in merged.columns else 'UNK')
        merged = merged.drop(columns=['_pn', '_pos', '_team'], errors='ignore')
        if 'years_exp' in merged.columns:
            merged['years_exp'] = merged['years_exp'].fillna(0)
        # Use 2024 rosters (team in target year) if available so team_strength = strength of team they're on in 2024
        rosters_2024_path = os.path.join(data_dir, "rosters_2024.csv")
        if os.path.isfile(rosters_2024_path):
            rosters_2024 = pd.read_csv(rosters_2024_path)
            rosters_2024['player_id'] = rosters_2024['player_id'].astype(str)
            rosters_2024 = rosters_2024[['player_id', 'team']].drop_duplicates(subset=['player_id'], keep='first').rename(columns={'team': 'team_target'})
            merged = merged.merge(rosters_2024, on='player_id', how='left')
            merged['team'] = merged['team_target'].fillna(merged['team'])
            merged = merged.drop(columns=['team_target'], errors='ignore')
        merged = merged.merge(teams[['team', 'name']], on='team', how='left')
        # Prior-year PPG (strong signal for next-year level)
        merged['prior_ppg'] = (merged['fantasy_points_prev'] / 17).fillna(0) if 'fantasy_points_prev' in merged.columns else 0
        # Features: 2023 stats (use _prev), position, team_strength, prior_ppg
        for col in FEATURE_COLUMNS:
            if col in ('position', 'team_strength', 'prior_ppg', 'years_exp'):
                continue
            prev_col = col + '_prev'
            merged[col] = merged[prev_col].fillna(0) if prev_col in merged.columns else 0
        if 'fantasy_points_prev' in merged.columns:
            merged['team_strength'] = merged.groupby('team')['fantasy_points_prev'].transform('mean').fillna(0) / 17
        else:
            merged['team_strength'] = 0
        position_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'K': 5, 'DEF': 6}
        merged['position'] = merged['position'].astype(str).map(position_map).fillna(0).astype(int)
        merged['target'] = merged['fantasy_points'] / 17
        for c in FEATURE_COLUMNS:
            if c not in merged.columns:
                merged[c] = 0
        player_data = merged[merged['target'] > 0].copy()
        if 'player_name' not in player_data.columns or player_data['player_name'].isna().all():
            player_data['player_name'] = player_data['player_id'].astype(str)
        logger.info("Forecast mode: 2023 features -> 2024 PPG (predict 2025 from 2024 at inference)")
    else:
        # Same-season mode: 2024 stats -> 2024 PPG (current behavior)
        player_data = players.merge(stats_2024, on='player_id', how='inner')
        player_data = player_data.merge(teams, on='team', how='left')
        player_data = player_data[
            (player_data['fantasy_points'] > 0) & (player_data['player_name'].notna())
        ].copy()
        player_data = _build_features_and_target(player_data)
        if 'years_exp' in player_data.columns:
            player_data['years_exp'] = player_data['years_exp'].fillna(0)
        for c in FEATURE_COLUMNS:
            if c not in player_data.columns:
                player_data[c] = 0
        logger.info("Same-season mode: 2024 stats -> 2024 PPG (add real_stats_2023.csv for forecast mode)")

    if player_data.empty or 'target' not in player_data.columns:
        logger.error("No training data.")
        return False

    # Sample weights: higher PPG players count more
    sample_weight = (1.0 + np.clip(player_data["target"].values, 0, SAMPLE_WEIGHT_MAX)).astype(np.float64)

    train_size = int(0.8 * len(player_data))
    train_data = player_data.iloc[:train_size]
    test_data = player_data.iloc[train_size:]
    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data['target']
    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data['target']
    sample_weight_train = sample_weight[:train_size]

    logger.info("Training on all players (n=%d) with sample weights (higher weight for higher PPG).", len(player_data))
    predictor = FantasyPredictor()
    predictor.train_models(X_train, y_train, sample_weight=sample_weight_train)
    predictions = predictor.predict(X_test)
    mae = float(np.mean(np.abs(predictions - y_test)))
    rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))
    r2 = float(1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - y_test.mean()) ** 2)))

    logger.info("  MAE: %.3f  RMSE: %.3f  R²: %.3f", mae, rmse, r2)
    joblib.dump(predictor, os.path.join(models_dir, "general_predictor.pkl"))

    summary = {
        'training_date': datetime.now().isoformat(),
        'forecast_mode': os.path.isfile(stats_2023_path),
        'target': '2024 PPG (actual); at inference we predict 2025 from 2024' if os.path.isfile(stats_2023_path) else '2024 PPG (same-season)',
        'sample_weighted': True,
        'n_train': len(player_data),
        'general_model_performance': {'mae': mae, 'rmse': rmse, 'r2': r2}
    }
    import json
    with open(os.path.join(models_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Models saved to %s", models_dir)
    return True

if __name__ == "__main__":
    success = train_models()
    
    if success:
        print("\n✅ Successfully trained models with real NFL data!")
        print("📁 Models saved to: models/")
        print("🎯 Ready for production predictions!")
    else:
        print("❌ Failed to train models with real data")
