"""
Fetch 2023, 2024, and 2025 NFL player stats from nflverse for prior-year forecasting.
Saves data/real_stats_2023.csv, real_stats_2024.csv, and (when available) real_stats_2025.csv.

Recommended (works on Python 3.13):
  pip install nflreadpy

If you're on older Python and nflreadpy fails:
  pip install nfl_data_py
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
_data_dir = os.path.join(_root, "data")
os.makedirs(_data_dir, exist_ok=True)


def main():
    seasons = [2023, 2024, 2025]
    df = None

    # Prefer nflreadpy (Python 3.12/3.13 compatible; no old pandas)
    try:
        import nflreadpy as nfl
        df = nfl.load_player_stats(seasons=seasons, summary_level="reg")
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        elif hasattr(df, "to_polars"):
            import pandas as pd
            df = pd.DataFrame(df)
    except ImportError:
        pass
    if df is None or df.empty:
        try:
            import nfl_data_py as nfl
            if hasattr(nfl, "load_player_stats"):
                df = nfl.load_player_stats(seasons)
            elif hasattr(nfl, "import_weekly_data"):
                df = nfl.import_weekly_data(seasons)
        except ImportError:
            pass

    if df is None or df.empty:
        print("Install nflreadpy (recommended on Python 3.13):")
        print("  pip install nflreadpy")
        print("Then run: python scripts/fetch_historical_nflverse.py")
        print("")
        print("Note: nfl_data_py often fails on Python 3.13 (old pandas). Use nflreadpy instead.")
        return

    # Normalize to our schema (nflverse uses various names)
    renames = {
        "player_display_name": "player_name",
        "recent_team": "team",
        "interceptions": "passing_ints",
        "passing_int": "passing_ints",
        "pass_yd": "passing_yards", "pass_td": "passing_tds", "pass_int": "passing_ints",
        "rush_yd": "rushing_yards", "rush_td": "rushing_tds",
        "rec_yd": "receiving_yards", "rec_td": "receiving_tds", "rec": "receptions",
        "attempts": "passing_attempts",
    }
    for old, new in renames.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    if "player_name" not in df.columns and "player_display_name" in df.columns:
        df["player_name"] = df["player_display_name"]
    if "passing_ints" not in df.columns and "interceptions" in df.columns:
        df["passing_ints"] = df["interceptions"]

    # If weekly, aggregate to season
    if "week" in df.columns and df["week"].notna().any():
        id_cols = ["player_id", "season"] if "season" in df.columns else ["player_id"]
        for c in ["player_name", "position", "recent_team", "team"]:
            if c in df.columns and c not in id_cols:
                id_cols.append(c)
        id_cols = [c for c in id_cols if c in df.columns]
        sum_cols = [c for c in [
            "passing_yards", "passing_tds", "passing_ints", "rushing_yards", "rushing_tds",
            "receiving_yards", "receiving_tds", "receptions", "fumbles", "fumbles_lost",
            "fantasy_points", "fantasy_points_ppr", "targets", "carries", "passing_attempts"
        ] if c in df.columns]
        if sum_cols:
            agg = {c: "sum" for c in sum_cols}
            first = {c: "first" for c in id_cols if c in df.columns and c not in sum_cols}
            for c in ["player_name", "position", "team", "recent_team"]:
                if c in df.columns and c not in sum_cols:
                    first[c] = "first"
            if "week" in df.columns:
                agg["games_played"] = ("week", "count")
            df = df.groupby([c for c in id_cols if c in df.columns], dropna=False).agg({**agg, **first}).reset_index()

    def _col(name, default=0):
        if name in df.columns:
            return df[name].fillna(0)
        try:
            import pandas as pd
            return pd.Series(default, index=df.index)
        except Exception:
            return default

    if "fantasy_points" not in df.columns:
        if "fantasy_points_ppr" in df.columns:
            df["fantasy_points"] = df["fantasy_points_ppr"]
        else:
            df["fantasy_points"] = (
                _col("passing_yards") * 0.04 + _col("passing_tds") * 4 - _col("passing_ints") * 2
                + _col("rushing_yards") * 0.1 + _col("rushing_tds") * 6
                + _col("receiving_yards") * 0.1 + _col("receiving_tds") * 6 + _col("receptions") * 1
            )
    if "fumbles" not in df.columns:
        df["fumbles"] = df["fumbles_lost"].fillna(0) if "fumbles_lost" in df.columns else 0
    if "week" not in df.columns:
        df["week"] = ""

    out = df[[c for c in [
        "player_id", "season", "week",
        "passing_yards", "passing_tds", "passing_ints", "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "receptions", "fumbles", "fantasy_points", "player_name",
        "position", "team", "games_played", "targets", "carries", "passing_attempts"
    ] if c in df.columns]].copy()
    out = out.rename(columns={})  # no-op, just ensure columns
    for c in ["passing_yards", "passing_tds", "passing_ints", "rushing_yards", "rushing_tds",
              "receiving_yards", "receiving_tds", "receptions", "fumbles", "fantasy_points"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(float)

    for season in seasons:
        if "season" in out.columns:
            sub = out[out["season"] == season]
        else:
            sub = out.copy()
            sub["season"] = season
        if sub.empty:
            print(f"No data for {season}; skipping real_stats_{season}.csv")
            continue
        path = os.path.join(_data_dir, f"real_stats_{season}.csv")
        sub.to_csv(path, index=False)
        print(f"Saved {len(sub)} rows → {path}")

    print("\n✅ Done. You have stats for prior-year → current-year forecasting (and 2025 actuals for evaluation when available).")


if __name__ == "__main__":
    main()
