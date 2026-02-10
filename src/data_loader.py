"""
Load players, teams, and stats from Redis cache, then Postgres, then CSV.
Used by both Streamlit app and FastAPI.
"""
import logging
import os
from typing import Optional, Tuple

import pandas as pd

from cache import (
    get_cached,
    set_cached,
    cache_key_players,
    cache_key_teams,
    cache_key_stats,
)
from db import get_players as db_players
from db import get_teams as db_teams
from db import get_player_stats as db_stats
from db import seed_from_csv_if_empty, is_available as db_available
from config import USE_DB

logger = logging.getLogger(__name__)


def _records_to_dataframe(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def load_data(data_dir: str = "data") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Returns (players, teams, stats, live_players, live_predictions).
    live_players and live_predictions are always from CSV if present (optional).
    """
    # Optional live data from CSV only
    live_players = None
    live_predictions = None
    if os.path.exists(os.path.join(data_dir, "live", "latest_players.csv")):
        try:
            live_players = pd.read_csv(os.path.join(data_dir, "live", "latest_players.csv"))
        except Exception as e:
            logger.debug("Could not load live_players: %s", e)
    if os.path.exists(os.path.join(data_dir, "live", "live_predictions.csv")):
        try:
            live_predictions = pd.read_csv(os.path.join(data_dir, "live", "live_predictions.csv"))
        except Exception as e:
            logger.debug("Could not load live_predictions: %s", e)

    if not USE_DB:
        return _load_from_csv(data_dir) + (live_players, live_predictions)

    # 1) Try cache
    try:
        p = get_cached(cache_key_players())
        t = get_cached(cache_key_teams())
        s = get_cached(cache_key_stats(2024))
        if p is not None and t is not None and s is not None:
            players = _records_to_dataframe(p)
            teams = _records_to_dataframe(t)
            stats = _records_to_dataframe(s)
            if not players.empty and not teams.empty:
                return players, teams, stats, live_players, live_predictions
    except Exception as e:
        logger.debug("Cache read failed: %s", e)

    # 2) Try Postgres (and seed from CSV if empty)
    if db_available():
        seed_from_csv_if_empty(data_dir)
        players = db_players()
        teams = db_teams()
        stats = db_stats(2024)
        if players is not None and not players.empty and teams is not None and not teams.empty:
            # Populate cache
            set_cached(cache_key_players(), players.to_dict(orient="records"))
            set_cached(cache_key_teams(), teams.to_dict(orient="records"))
            set_cached(cache_key_stats(2024), stats.to_dict(orient="records") if stats is not None and not stats.empty else [])
            return (
                players,
                teams,
                stats if stats is not None and not stats.empty else pd.DataFrame(),
                live_players,
                live_predictions,
            )

    # 3) Fallback to CSV
    return _load_from_csv(data_dir) + (live_players, live_predictions)


def _load_from_csv(data_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        players = pd.read_csv(os.path.join(data_dir, "real_players.csv"))
        teams = pd.read_csv(os.path.join(data_dir, "real_teams.csv"))
        stats = pd.read_csv(os.path.join(data_dir, "real_stats_2024.csv"))
        return players, teams, stats
    except FileNotFoundError as e:
        logger.warning("CSV fallback failed: %s", e)
        return None, None, None
    except Exception as e:
        logger.warning("CSV load failed: %s", e)
        return None, None, None
