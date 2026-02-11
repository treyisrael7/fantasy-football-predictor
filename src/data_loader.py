"""
Load players, teams, and stats from CSV. Used by the Streamlit app and optional API.
"""
import logging
import os
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(data_dir: str = "data") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Returns (players, teams, stats, live_players, live_predictions).
    live_players and live_predictions are optional CSV if present.
    """
    # Optional live data
    live_players = None
    live_predictions = None
    live_dir = os.path.join(data_dir, "live")
    if os.path.exists(os.path.join(live_dir, "latest_players.csv")):
        try:
            live_players = pd.read_csv(os.path.join(live_dir, "latest_players.csv"))
        except Exception as e:
            logger.debug("Could not load live_players: %s", e)
    if os.path.exists(os.path.join(live_dir, "live_predictions.csv")):
        try:
            live_predictions = pd.read_csv(os.path.join(live_dir, "live_predictions.csv"))
        except Exception as e:
            logger.debug("Could not load live_predictions: %s", e)

    players, teams, stats = _load_from_csv(data_dir)
    return players, teams, stats, live_players, live_predictions


def _load_from_csv(data_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        players = pd.read_csv(os.path.join(data_dir, "real_players.csv"))
        teams = pd.read_csv(os.path.join(data_dir, "real_teams.csv"))
        stats = pd.read_csv(os.path.join(data_dir, "real_stats_2024.csv"))
        return players, teams, stats
    except FileNotFoundError as e:
        logger.warning("CSV load failed (missing file): %s", e)
        return None, None, None
    except Exception as e:
        logger.warning("CSV load failed: %s", e)
        return None, None, None
