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


def _ensure_schema(conn):
    """Create tables if they don't exist (for Neon/empty Postgres). Only runs CREATE IF NOT EXISTS."""
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id VARCHAR(50) PRIMARY KEY,
                    player_name VARCHAR(100) NOT NULL,
                    position VARCHAR(10) NOT NULL,
                    team VARCHAR(10) NOT NULL,
                    status VARCHAR(20) DEFAULT 'Active',
                    height INTEGER, weight INTEGER, college VARCHAR(100), years_exp DECIMAL(3,1),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS teams (
                    team_id VARCHAR(10) PRIMARY KEY,
                    team VARCHAR(10) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    conference VARCHAR(10) NOT NULL,
                    division VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS player_stats (
                    id SERIAL PRIMARY KEY,
                    player_id VARCHAR(50) NOT NULL,
                    season INTEGER NOT NULL,
                    week INTEGER,
                    fantasy_points DECIMAL(6,2),
                    passing_yards INTEGER DEFAULT 0,
                    passing_tds INTEGER DEFAULT 0,
                    rushing_yards INTEGER DEFAULT 0,
                    rushing_tds INTEGER DEFAULT 0,
                    receiving_yards INTEGER DEFAULT 0,
                    receiving_tds INTEGER DEFAULT 0,
                    receptions INTEGER DEFAULT 0,
                    fumbles INTEGER DEFAULT 0,
                    interceptions INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_players_position ON players(position)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_stats_player_season ON player_stats(player_id, season)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    player_id VARCHAR(50) NOT NULL,
                    week INTEGER NOT NULL,
                    season INTEGER NOT NULL,
                    predicted_points DECIMAL(6,2) NOT NULL,
                    confidence DECIMAL(5,2),
                    model_version VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_predictions_week_season ON predictions(week, season)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS actual_results (
                    id SERIAL PRIMARY KEY,
                    player_id VARCHAR(50) NOT NULL,
                    week INTEGER NOT NULL,
                    season INTEGER NOT NULL,
                    actual_points DECIMAL(6,2) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, week, season)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_actuals_week_season ON actual_results(week, season)")
        conn.commit()
    except Exception as e:
        logger.debug("ensure_schema failed: %s", e)
        if conn:
            conn.rollback()


def get_players() -> Optional[pd.DataFrame]:
    conn = _get_connection()
    if not conn:
        return None
    _ensure_schema(conn)
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
    _ensure_schema(conn)
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
    _ensure_schema(conn)
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
    Creates tables automatically on Neon/empty DBs.
    """
    conn = _get_connection()
    if not conn:
        return False
    _ensure_schema(conn)
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


def save_predictions(week: int, season: int, predictions_df: pd.DataFrame, model_version: str = "v1") -> bool:
    """Save a batch of predictions to the DB. predictions_df must have player_id and projection (predicted points)."""
    conn = _get_connection()
    if not conn or predictions_df is None or predictions_df.empty:
        return False
    if "player_id" not in predictions_df.columns or "projection" not in predictions_df.columns:
        return False
    _ensure_schema(conn)
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM predictions WHERE week = %s AND season = %s", (week, season))
            for _, row in predictions_df.iterrows():
                try:
                    cur.execute(
                        """INSERT INTO predictions (player_id, week, season, predicted_points, confidence, model_version)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (
                            str(row["player_id"]),
                            week,
                            season,
                            float(row["projection"]),
                            float(row.get("confidence", 0)) if pd.notna(row.get("confidence")) else None,
                            model_version,
                        ),
                    )
                except Exception as e:
                    logger.debug("Skip prediction row %s: %s", row.get("player_id"), e)
        conn.commit()
        return True
    except Exception as e:
        logger.warning("save_predictions failed: %s", e)
        if conn:
            conn.rollback()
        return False


def get_saved_predictions(week: int, season: int) -> Optional[pd.DataFrame]:
    """Return saved predictions for a week/season, with player_id, predicted_points, confidence."""
    conn = _get_connection()
    if not conn:
        return None
    _ensure_schema(conn)
    try:
        return pd.read_sql(
            "SELECT player_id, week, season, predicted_points, confidence FROM predictions WHERE week = %s AND season = %s",
            conn,
            params=(week, season),
        )
    except Exception as e:
        logger.debug("get_saved_predictions failed: %s", e)
        return None


def save_actuals(week: int, season: int, actuals: list) -> bool:
    """Save actual results. actuals = list of dicts with player_id and actual_points."""
    conn = _get_connection()
    if not conn or not actuals:
        return False
    _ensure_schema(conn)
    try:
        with conn.cursor() as cur:
            for row in actuals:
                try:
                    cur.execute(
                        """INSERT INTO actual_results (player_id, week, season, actual_points)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (player_id, week, season) DO UPDATE SET actual_points = EXCLUDED.actual_points""",
                        (str(row["player_id"]), week, season, float(row["actual_points"])),
                    )
                except Exception as e:
                    logger.debug("Skip actual row: %s", e)
        conn.commit()
        return True
    except Exception as e:
        logger.warning("save_actuals failed: %s", e)
        if conn:
            conn.rollback()
        return False


def get_actuals(week: int, season: int) -> Optional[pd.DataFrame]:
    conn = _get_connection()
    if not conn:
        return None
    _ensure_schema(conn)
    try:
        return pd.read_sql(
            "SELECT player_id, week, season, actual_points FROM actual_results WHERE week = %s AND season = %s",
            conn,
            params=(week, season),
        )
    except Exception as e:
        logger.debug("get_actuals failed: %s", e)
        return None


def get_accuracy(week: Optional[int], season: Optional[int]) -> Optional[dict]:
    """
    Join predictions and actual_results, compute MAE/RMSE. If week/season None, use all saved data.
    Returns dict with mae, rmse, n, by_position (optional), rows (DataFrame of predicted vs actual).
    """
    conn = _get_connection()
    if not conn:
        return None
    _ensure_schema(conn)
    try:
        if week is not None and season is not None:
            sql = """
                SELECT p.player_id, p.predicted_points, a.actual_points
                FROM predictions p
                INNER JOIN actual_results a ON p.player_id = a.player_id AND p.week = a.week AND p.season = a.season
                WHERE p.week = %s AND p.season = %s
            """
            params = (week, season)
        elif season is not None:
            sql = """
                SELECT p.player_id, p.week, p.season, p.predicted_points, a.actual_points
                FROM predictions p
                INNER JOIN actual_results a ON p.player_id = a.player_id AND p.week = a.week AND p.season = a.season
                WHERE p.season = %s
            """
            params = (season,)
        else:
            sql = """
                SELECT p.player_id, p.week, p.season, p.predicted_points, a.actual_points
                FROM predictions p
                INNER JOIN actual_results a ON p.player_id = a.player_id AND p.week = a.week AND p.season = a.season
            """
            params = ()
        df = pd.read_sql(sql, conn, params=params)
        if df.empty:
            return {"mae": None, "rmse": None, "n": 0, "rows": df}
        err = df["predicted_points"].astype(float) - df["actual_points"].astype(float)
        mae = float(err.abs().mean())
        rmse = float((err ** 2).mean() ** 0.5)
        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "n": len(df), "rows": df}
    except Exception as e:
        logger.debug("get_accuracy failed: %s", e)
        return None
