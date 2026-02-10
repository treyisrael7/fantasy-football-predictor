"""
App config from environment. Used by web app, API, and scripts.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL: use DATABASE_URL or build from parts (Docker Compose style)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    os.getenv(
        "POSTGRES_URL",
        "postgresql://fantasy_user:fantasy_password@localhost:5432/fantasy_football",
    ),
)

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# When True, app will try Postgres/Redis first and fall back to CSV if unavailable
USE_DB = os.getenv("USE_DB", "false").lower() in ("1", "true", "yes")
