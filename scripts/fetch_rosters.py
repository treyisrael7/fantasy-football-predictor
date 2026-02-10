"""
Fetch 2024 and 2025 NFL rosters from nflverse (nflreadpy).
Saves data/rosters_2024.csv and data/rosters_2025.csv (player_id, team) so we can use
"next year's team" when predicting (2024 roster for 2023->2024 training, 2025 roster for 2025 app predictions).

Requires: pip install nflreadpy
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
_data_dir = os.path.join(_root, "data")
os.makedirs(_data_dir, exist_ok=True)


def main():
    try:
        import nflreadpy as nfl
    except ImportError:
        print("Install nflreadpy: pip install nflreadpy")
        return

    seasons = [2024, 2025]
    for season in seasons:
        try:
            rosters = nfl.load_rosters(seasons=season)
            if hasattr(rosters, "to_pandas"):
                rosters = rosters.to_pandas()
            # nflverse uses gsis_id; we save as player_id to match our stats
            if "gsis_id" in rosters.columns:
                rosters = rosters.rename(columns={"gsis_id": "player_id"})
            if "player_id" not in rosters.columns:
                print(f"No player_id/gsis_id in {season} rosters, skipping.")
                continue
            # One row per player per season (take first week if multiple); include position when available
            cols = ["player_id", "team"]
            if "position" in rosters.columns:
                cols.append("position")
            if "season" in rosters.columns:
                cols.append("season")
            cols = [c for c in cols if c in rosters.columns]
            out = rosters[cols].drop_duplicates(subset=["player_id"], keep="first")
            path = os.path.join(_data_dir, f"rosters_{season}.csv")
            out.to_csv(path, index=False)
            print(f"Saved {len(out)} rows -> {path}")
        except Exception as e:
            print(f"Could not load {season} rosters: {e}")

    print("\nDone. Use rosters_2024 for training (team in target year), rosters_2025 for app (2025 predictions).")


if __name__ == "__main__":
    main()
