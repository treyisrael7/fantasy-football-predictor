"""
Sync CSV data into PostgreSQL. Run after data_collection.py to load/refresh DB.
Usage (from project root):
  python scripts/sync_to_postgres.py
  # or with env:
  DATABASE_URL=postgresql://... python scripts/sync_to_postgres.py
"""
import os
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from db import seed_from_csv_if_empty, is_available

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not is_available():
        print("PostgreSQL is not available (check DATABASE_URL).")
        sys.exit(1)
    if seed_from_csv_if_empty(data_dir=data_dir):
        print("Database synced from CSV (or already had data).")
    else:
        print("Sync failed or CSV files missing in data/.")
        sys.exit(1)

if __name__ == "__main__":
    main()
