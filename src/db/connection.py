"""
PostgreSQL data access. Returns DataFrames for players, teams, and stats.
"""
import logging
import os
from typing import Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

_connection = None


def _parse_db_url(url: str) -> dict:
    """Convert postgresql:// URL to psycopg2 connect kwargs."""
    parsed = urlparse(url)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": (parsed.path or "/fantasy_football").lstrip("/") or "fantasy_football",
        "user": parsed.username or "fantasy_user",
        "password": parsed.password or "fantasy_password",
    }


def _get_connection():
    global _connection
    if _connection is not None:
        try:
            with _connection.cursor() as cur:
                cur.execute("SELECT 1")
            return _connection
        except Exception:
            _connection = None
    try:
        import psycopg2
        url = os.getenv(
            "DATABASE_URL",
            os.getenv("POSTGRES_URL", "postgresql://fantasy_user:fantasy_password@localhost:5432/fantasy_football"),
        )
        kwargs = _parse_db_url(url)
        _connection = psycopg2.connect(**kwargs)
        return _connection
    except Exception as e:
        logger.debug("Postgres not available: %s", e)
        return None


def is_available() -> bool:
    return _get_connection() is not None


def get_players() -> Optional[pd.DataFrame]:
    conn = _get_connection()
    if not conn:
        return None
    try:
        return pd.read_sql(
            "SELECT player_id, player_name, position, team, status, height, weight, college, years_exp FROM players",
            conn,
        )
    except Exception as e:
        logger.debug("get_players failed: %s", e)
        return None


def get_teams() -> Optional[pd.DataFrame]:
    conn = _get_connection()
    if not conn:
        return None
    try:
        return pd.read_sql(
            "SELECT team_id, team, name, conference, division FROM teams",
            conn,
        )
    except Exception as e:
        logger.debug("get_teams failed: %s", e)
        return None


def get_player_stats(season: int = 2024) -> Optional[pd.DataFrame]:
    conn = _get_connection()
    if not conn:
        return None
    try:
        df = pd.read_sql(
            """
            SELECT player_id, season, week, fantasy_points,
                   passing_yards, passing_tds, interceptions AS passing_ints,
                   rushing_yards, rushing_tds, receiving_yards, receiving_tds,
                   receptions, fumbles
            FROM player_stats
            WHERE season = %s
            """,
            conn,
            params=(season,),
        )
        return df
    except Exception as e:
        logger.debug("get_player_stats failed: %s", e)
        return None


def _safe_int(x):
    """Return int(x) or None for NaN/empty."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return None


def _stats_columns_for_insert():
    return [
        "player_id", "season", "week", "fantasy_points",
        "passing_yards", "passing_tds", "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "receptions", "fumbles", "interceptions",
    ]


def seed_from_csv_if_empty(data_dir: str = "data") -> bool:
    """
    If players table is empty, load real_players.csv, real_teams.csv, real_stats_2024.csv
    into Postgres. Returns True if seeding was performed or data already existed.
    """
    conn = _get_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM players")
            if cur.fetchone()[0] > 0:
                return True  # already seeded
    except Exception as e:
        logger.debug("seed check failed: %s", e)
        return False

    players_path = os.path.join(data_dir, "real_players.csv")
    teams_path = os.path.join(data_dir, "real_teams.csv")
    stats_path = os.path.join(data_dir, "real_stats_2024.csv")
    if not all(os.path.isfile(p) for p in (players_path, teams_path, stats_path)):
        logger.warning("CSV files not found for seeding: %s", data_dir)
        return False

    try:
        teams = pd.read_csv(teams_path)
        players = pd.read_csv(players_path)
        stats = pd.read_csv(stats_path)
    except Exception as e:
        logger.warning("Failed to read CSV for seeding: %s", e)
        return False

    try:
        with conn.cursor() as cur:
            # Teams (upsert by team_id)
            for _, row in teams.iterrows():
                cur.execute(
                    """
                    INSERT INTO teams (team_id, team, name, conference, division)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (team_id) DO UPDATE SET
                      team = EXCLUDED.team, name = EXCLUDED.name,
                      conference = EXCLUDED.conference, division = EXCLUDED.division
                    """,
                    (
                        str(row.get("team_id", row.get("team", ""))),
                        str(row.get("team", "")),
                        str(row.get("name", "")),
                        str(row.get("conference", "")),
                        str(row.get("division", "")),
                    ),
                )
            # Players (upsert by player_id)
            for _, row in players.iterrows():
                cur.execute(
                    """
                    INSERT INTO players (player_id, player_name, position, team, status, height, weight, college, years_exp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id) DO UPDATE SET
                      player_name = EXCLUDED.player_name, position = EXCLUDED.position, team = EXCLUDED.team,
                      status = EXCLUDED.status, height = EXCLUDED.height, weight = EXCLUDED.weight,
                      college = EXCLUDED.college, years_exp = EXCLUDED.years_exp, updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        str(row["player_id"]),
                        str(row.get("player_name", "")),
                        str(row.get("position", "")),
                        str(row.get("team", "")),
                        str(row.get("status", "Active")),
                        pd.to_numeric(row.get("height"), errors="coerce") or None,
                        pd.to_numeric(row.get("weight"), errors="coerce") or None,
                        str(row.get("college", "")) if pd.notna(row.get("college")) else None,
                        pd.to_numeric(row.get("years_exp"), errors="coerce") or None,
                    ),
                )
            # Player stats (insert; avoid duplicate key by using conflict skip or delete first)
            cur.execute("DELETE FROM player_stats WHERE season = %s", (int(stats["season"].iloc[0]) if "season" in stats.columns else 2024,))
            cols = _stats_columns_for_insert()
            for _, row in stats.iterrows():
                try:
                    cur.execute(
                        """
                        INSERT INTO player_stats (player_id, season, week, fantasy_points,
                          passing_yards, passing_tds, rushing_yards, rushing_tds,
                          receiving_yards, receiving_tds, receptions, fumbles, interceptions)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(row["player_id"]),
                            int(row.get("season", 2024)),
                            _safe_int(row.get("week")),
                            float(row.get("fantasy_points", 0)),
                            int(row.get("passing_yards", 0) or 0),
                            int(row.get("passing_tds", 0) or 0),
                            int(row.get("rushing_yards", 0) or 0),
                            int(row.get("rushing_tds", 0) or 0),
                            int(row.get("receiving_yards", 0) or 0),
                            int(row.get("receiving_tds", 0) or 0),
                            int(row.get("receptions", 0) or 0),
                            int(row.get("fumbles", 0) or 0),
                            int(row.get("passing_ints", row.get("interceptions", 0)) or 0),
                        ),
                    )
                except Exception as e:
                    logger.debug("Skip stat row %s: %s", row.get("player_id"), e)
                    continue
        conn.commit()
        logger.info("Seeded database from CSV")
        return True
    except Exception as e:
        logger.warning("Seeding failed: %s", e)
        if conn:
            conn.rollback()
        return False
