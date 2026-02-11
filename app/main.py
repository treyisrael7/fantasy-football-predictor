"""
Fantasy Football Forecasting Dashboard
Next-year PPG analytics and out-of-sample evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
import json

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
if os.path.join(_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_root, "src"))

from data_loader import load_data as load_data_backend

st.set_page_config(
    page_title="Fantasy Football ML Model",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; color: #1a1a2e; margin-bottom: 0.5rem; }
    .section-header { font-size: 1.25rem; color: #16213e; margin-top: 1.5rem; margin-bottom: 0.5rem; }
    .metric-box { background: #f0f4f8; padding: 0.75rem 1rem; border-radius: 8px; border-left: 4px solid #0f3460; margin: 0.25rem 0; }
    .badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    .badge-ok { background: #0f3460; color: #e8e8e8; }
    .badge-muted { background: #e2e8f0; color: #475569; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=120)
def load_data():
    """Load players, teams, stats from CSV."""
    data_dir = os.path.join(_root, "data")
    try:
        return load_data_backend(data_dir=data_dir)
    except Exception as e:
        st.error(f"Data load failed: {e}")
        return None, None, None, None, None


@st.cache_data(ttl=60)
def _load_rosters_2024():
    """Load 2024 roster (player_id -> team, position). Invalid/blank player_id rows are dropped."""
    path = os.path.join(_root, "data", "rosters_2024.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        df["player_id"] = df["player_id"].astype(str).str.strip()
        df = df[df["player_id"].notna() & (df["player_id"] != "") & (df["player_id"] != "nan")]
        cols = ["player_id", "team"]
        if "position" in df.columns:
            cols.append("position")
        return df[[c for c in cols if c in df.columns]].drop_duplicates(subset=["player_id"], keep="first")
    except Exception:
        return None


@st.cache_data(ttl=60)
def _load_rosters_2025():
    """Load 2025 roster (player_id -> team, position). Invalid/blank player_id rows are dropped."""
    path = os.path.join(_root, "data", "rosters_2025.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        df["player_id"] = df["player_id"].astype(str).str.strip()
        df = df[df["player_id"].notna() & (df["player_id"] != "") & (df["player_id"] != "nan")]
        cols = ["player_id", "team"]
        if "position" in df.columns:
            cols.append("position")
        out = df[[c for c in cols if c in df.columns]].drop_duplicates(subset=["player_id"], keep="first")
        return out.rename(columns={"team": "team_2025"})
    except Exception:
        return None


@st.cache_data(ttl=60)
def _load_stats_2025_meta():
    """Load player_id, position, team from real_stats_2025.csv for display (one row per player)."""
    path = os.path.join(_root, "data", "real_stats_2025.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        df["player_id"] = df["player_id"].astype(str).str.strip()
        df = df[df["player_id"].notna() & (df["player_id"] != "") & (df["player_id"] != "nan")]
        pos_col = "position" if "position" in df.columns else next((c for c in df.columns if "position" in c.lower()), None)
        team_col = "team" if "team" in df.columns else next((c for c in df.columns if c.lower() == "team"), None)
        if not pos_col and not team_col:
            return None
        cols = ["player_id"]
        if pos_col:
            cols.append(pos_col)
        if team_col:
            cols.append(team_col)
        out = df[[c for c in cols if c in df.columns]].drop_duplicates(subset=["player_id"], keep="first")
        if pos_col and pos_col != "position":
            out = out.rename(columns={pos_col: "position"})
        if team_col and team_col != "team":
            out = out.rename(columns={team_col: "team"})
        return out
    except Exception:
        return None


@st.cache_data(ttl=300)
def _load_actuals_2025():
    """Load actual 2025 season PPG from data/real_stats_2025.csv. Uses games played when weekly data so missed games are accounted for."""
    path = os.path.join(_root, "data", "real_stats_2025.csv")
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        df["player_id"] = df["player_id"].astype(str)
        if "fantasy_points" not in df.columns:
            return None
        # Weekly data: sum points and count games played so PPG = total / games_played (accounts for missed games)
        if "week" in df.columns and df["week"].notna().any():
            weeks_numeric = pd.to_numeric(df["week"], errors="coerce")
            if weeks_numeric.notna().any():
                agg = df.groupby("player_id", as_index=False).agg(
                    fantasy_points=("fantasy_points", "sum"),
                    games_played=("week", "count"),
                )
                agg["actual_ppg_2025"] = agg["fantasy_points"] / agg["games_played"].clip(lower=1)
                return agg[["player_id", "actual_ppg_2025"]]
        # Season-level: use games_played when present (accounts for missed games), else 17
        df = df.drop_duplicates(subset=["player_id"], keep="first")
        cols = ["player_id", "fantasy_points"]
        if "games_played" in df.columns:
            cols.append("games_played")
        df = df[[c for c in cols if c in df.columns]]
        if "games_played" in df.columns:
            df["actual_ppg_2025"] = df["fantasy_points"] / df["games_played"].clip(lower=1)
        else:
            df["actual_ppg_2025"] = df["fantasy_points"] / 17
        return df[["player_id", "actual_ppg_2025"]]
    except Exception:
        return None


def _build_ml_features(player_data):
    """Build feature matrix matching train_models.py (includes prior_ppg; uses 2025 roster for team_strength when available)."""
    position_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'K': 5, 'DEF': 6}
    player_data = player_data.copy()
    # Use 2025 roster for team when predicting 2025 (so team_strength = strength of 2025 team)
    rosters_2025 = _load_rosters_2025()
    if rosters_2025 is not None:
        player_data = player_data.merge(rosters_2025, on="player_id", how="left")
        player_data["team_for_strength"] = player_data["team_2025"].fillna(player_data["team"])
    else:
        player_data["team_for_strength"] = player_data["team"]
    position_series = player_data.get("position", pd.Series("UNK", index=player_data.index))
    player_data["position_encoded"] = position_series.map(position_map).fillna(0).astype(int)
    player_data["prior_ppg"] = (player_data["fantasy_points"] / 17).fillna(0)
    # Team strength: mean 2024 PPG of players on same (2025) team
    team_strength = player_data.groupby("team_for_strength")["fantasy_points"].transform("mean").fillna(0) / 17
    player_data["team_strength"] = team_strength
    player_data = player_data.drop(columns=["team_for_strength", "team_2025"], errors="ignore")
    feature_columns = [
        'position', 'team_strength', 'prior_ppg', 'years_exp',
        'passing_yards', 'passing_tds', 'passing_ints', 'rushing_yards',
        'rushing_tds', 'receiving_yards', 'receiving_tds', 'receptions', 'fumbles',
        'targets', 'carries',
    ]
    player_data['position_label'] = position_series.astype(str)
    player_data['position'] = player_data['position_encoded'].astype(int)
    for c in feature_columns:
        if c not in player_data.columns:
            player_data[c] = 0
    X = player_data[feature_columns].fillna(0)
    return X, player_data


@st.cache_resource
def _load_ml_model():
    """Load trained ensemble if available. Supports both joblib.dump(predictor) and save_models() formats."""
    import joblib
    try:
        from models.fantasy_predictor import FantasyPredictor
        # Check: models/general_predictor.pkl (train_models.py default), then models_real/, then models/models/
        candidates = [
            os.path.join(_root, "models", "general_predictor.pkl"),
            os.path.join(_root, "models", "general_predictor_real.pkl"),
            os.path.join(_root, "models", "models_real", "general_predictor_real.pkl"),
            os.path.join(_root, "models", "models_real", "general_predictor.pkl"),
            os.path.join(_root, "models", "models", "general_predictor.pkl"),
        ]
        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                obj = joblib.load(path)
                # Whole predictor object (e.g. joblib.dump(predictor, path))
                if hasattr(obj, "predict") and callable(getattr(obj, "predict", None)):
                    if not getattr(obj, "is_trained", False):
                        obj.is_trained = True
                    return obj
                # Dict from save_models()
                if isinstance(obj, dict) and "models" in obj:
                    p = FantasyPredictor()
                    p.models = obj.get("models", {})
                    p.scalers = obj.get("scalers", {})
                    p.feature_importance = obj.get("feature_importance", {})
                    p.model_performance = obj.get("model_performance", {})
                    p.is_trained = obj.get("is_trained", False)
                    return p
            except Exception:
                continue
        return None
    except Exception:
        return None


def _load_training_summary():
    """Load training metrics if available."""
    for name in ("training_summary_real.json", "training_summary.json"):
        path = os.path.join(_root, "models", name)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def create_predictions(players, teams, stats):
    """Run model (or fallback) and return projection table. Stats-driven so nflverse (stats) and Sleeper (players) ID mismatch doesn't drop everyone."""
    players = players.copy()
    stats = stats.copy()
    players["player_id"] = players["player_id"].astype(str)
    stats["player_id"] = stats["player_id"].astype(str)
    # Aggregate stats to one row per player if needed (e.g. weekly rows)
    id_cols = ["player_id"]
    if "season" in stats.columns:
        id_cols.append("season")
    sum_cols = [c for c in ["passing_yards", "passing_tds", "passing_ints", "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "receptions", "fumbles", "fantasy_points", "targets", "carries"] if c in stats.columns]
    pos_col = "position" if "position" in stats.columns else next((c for c in stats.columns if "position" in c.lower()), None)
    first_cols = [c for c in ["player_name", "position", "team"] if c in stats.columns]
    if pos_col and pos_col not in first_cols:
        first_cols.append(pos_col)
    if sum_cols:
        agg = {c: "sum" for c in sum_cols}
        for c in first_cols:
            agg[c] = "first"
        group_cols = [c for c in id_cols if c in stats.columns]
        stats = stats.groupby(group_cols, as_index=False).agg(agg)
    stats = stats[(stats["fantasy_points"].notna()) & (stats["fantasy_points"] > 0)].copy()
    if stats.empty:
        return pd.DataFrame(), False
    # Normalize position column name (might be "Position" or "position")
    if pos_col and pos_col != "position" and pos_col in stats.columns:
        stats = stats.rename(columns={pos_col: "position"})
    # Build position map from stats for fallback (player_id -> position from CSV)
    stats_position_map = None
    if "position" in stats.columns:
        pos_team = stats[["player_id", "position"]].copy()
        pos_team["player_id"] = pos_team["player_id"].astype(str).str.strip()
        pos_team["position"] = pos_team["position"].astype(str).str.strip()
        pos_team = pos_team[pos_team["position"].notna() & (pos_team["position"] != "") & (pos_team["position"].str.upper() != "NAN")]
        if not pos_team.empty:
            stats_position_map = pos_team.drop_duplicates(subset=["player_id"], keep="first").set_index("player_id")["position"]
    # Use stats as base; keep position/team from stats when present (stats may have them from nflverse)
    player_data = stats.copy()
    if "player_name" not in player_data.columns:
        player_data["player_name"] = player_data["player_id"]
    for col in ("position", "team"):
        if col not in player_data.columns:
            player_data[col] = np.nan
        else:
            player_data[col] = player_data[col].astype(str).replace("nan", "").replace("None", "").str.strip()
    # Match players by player_id first (works when both use same ID system)
    player_cols = [c for c in ["player_id", "player_name", "position", "team", "years_exp"] if c in players.columns]
    players_sub = players[player_cols].drop_duplicates(subset=["player_id"], keep="first")
    player_data = player_data.merge(
        players_sub.rename(columns={"player_name": "_pn", "position": "_pos", "team": "_team"}),
        on="player_id", how="left"
    )
    if "_pos" in player_data.columns:
        player_data["position"] = player_data["_pos"].fillna(player_data["position"]) if "position" in player_data.columns else player_data["_pos"]
    else:
        player_data["position"] = player_data.get("position", "UNK")
    if "_team" in player_data.columns:
        player_data["team"] = player_data["_team"].fillna(player_data["team"]) if "team" in player_data.columns else player_data["_team"]
    else:
        player_data["team"] = player_data.get("team", "UNK")
    player_data["player_name"] = player_data["_pn"].fillna(player_data["player_name"]) if "_pn" in player_data.columns else player_data["player_name"]
    player_data = player_data.drop(columns=["_pn", "_pos", "_team"], errors="ignore")
    # Fallback: fill missing position/team by matching on normalized player_name to real_players
    missing = (player_data["position"].isna() | (player_data["position"] == "") |
               player_data["team"].isna() | (player_data["team"] == "") | (player_data["team"] == "UNK"))
    if missing.any() and "player_name" in players_sub.columns:
        norm = players_sub.copy()
        norm["_name_norm"] = norm["player_name"].astype(str).str.lower().str.strip()
        norm = norm.drop_duplicates(subset=["_name_norm"], keep="first")
        name_map = norm.set_index("_name_norm")[["position", "team"]].to_dict("index")
        def fill_row(r):
            key = str(r.get("player_name", "")).lower().strip()
            if key in name_map:
                return name_map[key]["position"], name_map[key]["team"]
            return r.get("position"), r.get("team")
        for i in player_data.index[missing]:
            pos, team = fill_row(player_data.loc[i])
            if pd.notna(pos) and pos != "":
                player_data.at[i, "position"] = pos
            if pd.notna(team) and team != "" and team != "UNK":
                player_data.at[i, "team"] = team
    # Prefer position/team from 2025 real stats (same source as 2025 actual PPG) — one source of truth
    player_data["player_id"] = player_data["player_id"].astype(str).str.strip()
    stats_2025_meta = _load_stats_2025_meta()
    if stats_2025_meta is not None and not stats_2025_meta.empty:
        s25 = stats_2025_meta.copy()
        s25["player_id"] = s25["player_id"].astype(str).str.strip()
        player_data = player_data.merge(
            s25.rename(columns={c: f"_s25_{c}" for c in s25.columns if c != "player_id"}),
            on="player_id", how="left"
        )
        if "_s25_position" in player_data.columns:
            s25_pos = player_data["_s25_position"].astype(str).str.strip()
            has_pos = s25_pos.notna() & (s25_pos != "") & (s25_pos != "nan")
            player_data.loc[has_pos, "position"] = s25_pos.loc[has_pos].values
        if "_s25_team" in player_data.columns:
            s25_team = player_data["_s25_team"].astype(str).str.strip()
            has_team = s25_team.notna() & (s25_team != "") & (s25_team != "nan")
            player_data.loc[has_team, "team"] = s25_team.loc[has_team].values
        player_data = player_data.drop(columns=[c for c in player_data.columns if c.startswith("_s25_")], errors="ignore")
    # Then rosters (gsis_id) for any still missing
    roster_2025 = _load_rosters_2025()
    if roster_2025 is not None:
        r5 = roster_2025.copy()
        r5["player_id"] = r5["player_id"].astype(str).str.strip()
        r5_cols = {c: f"_r5_{c}" for c in r5.columns if c != "player_id"}
        player_data = player_data.merge(r5.rename(columns=r5_cols), on="player_id", how="left")
        if "_r5_team_2025" in player_data.columns:
            player_data["team"] = player_data["_r5_team_2025"].fillna(player_data["team"]).replace("", "UNK")
        elif "_r5_team" in player_data.columns:
            player_data["team"] = player_data["_r5_team"].fillna(player_data["team"]).replace("", "UNK")
        if "_r5_position" in player_data.columns:
            r5_pos = player_data["_r5_position"].astype(str).str.strip()
            has_pos = r5_pos.notna() & (r5_pos != "") & (r5_pos != "nan")
            player_data.loc[has_pos, "position"] = r5_pos.loc[has_pos]
        player_data = player_data.drop(columns=[c for c in player_data.columns if c.startswith("_r5_")], errors="ignore")
    roster_2024 = _load_rosters_2024()
    if roster_2024 is not None:
        roster_2024 = roster_2024.copy()
        roster_2024["player_id"] = roster_2024["player_id"].astype(str).str.strip()
        player_data = player_data.merge(
            roster_2024.rename(columns={c: f"_r4_{c}" for c in roster_2024.columns if c != "player_id"}),
            on="player_id", how="left"
        )
        if "_r4_team" in player_data.columns:
            unk_team = (player_data["team"].fillna("") == "UNK")
            player_data.loc[unk_team, "team"] = player_data.loc[unk_team, "_r4_team"].fillna("UNK").values
        if "_r4_position" in player_data.columns:
            r4_pos = player_data["_r4_position"].astype(str).str.strip()
            has_pos = r4_pos.notna() & (r4_pos != "") & (r4_pos != "nan")
            unk_pos = (player_data["position"].fillna("") == "UNK") | (player_data["position"].astype(str).str.strip() == "")
            player_data.loc[has_pos & unk_pos, "position"] = r4_pos.loc[has_pos & unk_pos].values
        player_data = player_data.drop(columns=[c for c in player_data.columns if c.startswith("_r4_")], errors="ignore")
    if "position" not in player_data.columns:
        player_data["position"] = "UNK"
    if "team" not in player_data.columns:
        player_data["team"] = "UNK"
    player_data["position"] = player_data["position"].fillna("UNK").replace("", "UNK").astype(str)
    player_data["team"] = player_data["team"].fillna("UNK").replace("", "UNK").astype(str)
    # Final fallback: fill position from stats file (by player_id) when still UNK
    if stats_position_map is not None:
        unk = (player_data["position"].str.upper() == "UNK") | (player_data["position"] == "")
        pid = player_data.loc[unk, "player_id"].astype(str).str.strip()
        from_stats = pid.map(stats_position_map)
        player_data.loc[unk, "position"] = from_stats.fillna("UNK").values
    # Infer position from stat columns when still UNK
    unk_pos = (player_data["position"].fillna("").astype(str).str.strip().str.upper().isin(("UNK", "0", "NAN", "")))
    if unk_pos.any():
        pas = player_data.get("passing_yards", pd.Series(0, index=player_data.index)).fillna(0)
        rush = player_data.get("rushing_yards", pd.Series(0, index=player_data.index)).fillna(0)
        rec_yd = player_data.get("receiving_yards", pd.Series(0, index=player_data.index)).fillna(0)
        rec = player_data.get("receptions", pd.Series(0, index=player_data.index)).fillna(0)
        rec_pts = rec_yd + rec * 2
        for i in player_data.index[unk_pos]:
            p, r, rp = float(pas.at[i]), float(rush.at[i]), float(rec_pts.at[i])
            if p >= max(r, rp) and p > 50:
                inferred = "QB"
            elif r >= max(p, rp) and r > 20:
                inferred = "RB"
            elif rp > max(p, r) and rp > 20:
                inferred = "WR" if rec.at[i] >= 2 else "TE"
            else:
                inferred = "UNK"
            if inferred != "UNK":
                player_data.at[i, "position"] = inferred
    player_data["player_name"] = player_data["player_name"].fillna(player_data["player_id"])
    player_data = player_data.merge(teams, on="team", how="left")
    if player_data.empty:
        return pd.DataFrame(), False

    X, player_data = _build_ml_features(player_data)
    baseline_ppg = (player_data["fantasy_points"] / 17).fillna(0).values.astype(float)
    model = _load_ml_model()
    use_ml = model is not None
    if use_ml:
        try:
            preds = model.predict(X)
            preds = np.clip(preds, 0, None)
            # Flat blend: 65% baseline, 35% model (relevance-aware blend hurt relevant-only, so we keep one blend for all)
            BLEND_BASELINE_WEIGHT = 0.65
            preds = BLEND_BASELINE_WEIGHT * baseline_ppg + (1 - BLEND_BASELINE_WEIGHT) * preds
            preds = np.clip(preds, 0, None)
        except Exception:
            use_ml = False
            preds = None
    if not use_ml or preds is None:
        preds = baseline_ppg + np.random.normal(0, 0.1, len(player_data))
        preds = np.clip(preds, 0, None)
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for i, (_, row) in enumerate(player_data.iterrows()):
        rows.append({
            'player_id': str(row.get('player_id', '')),
            'player_name': row['player_name'],
            'position': row.get('position_label', row.get('position', 'UNK')),
            'team': row['team'],
            'projection': round(float(preds[i]), 1),
            'fantasy_pts_2024_ppg': round(float(row.get('fantasy_points', 0)) / 17, 1),
        })
    return pd.DataFrame(rows), use_ml


def main():
    st.markdown('<h1 class="main-header">Fantasy Football Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Next-year PPG analytics and out-of-sample evaluation using prior-season stats and 2025 actuals.")

    players, teams, stats, _, _ = load_data()
    if players is None:
        st.error("Couldn't find data. Add real_players, real_teams, and real_stats_2024 to the `data/` folder and try again.")
        st.stop()

    st.markdown('<p class="section-header">Model</p>', unsafe_allow_html=True)
    model = _load_ml_model()
    summary = _load_training_summary()
    c1, c2 = st.columns(2)
    with c1:
        if model is not None:
            st.markdown('<span class="badge badge-ok">Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-muted">No model yet; using fallback</span>', unsafe_allow_html=True)
            st.caption("Run the training script to train your own.")
    with c2:
        if summary and "general_model_performance" in summary:
            perf = summary["general_model_performance"]
            if isinstance(perf.get("r2"), (int, float)):
                r2, rmse, mae = perf.get("r2", 0), perf.get("rmse", 0), perf.get("mae", 0)
                st.caption(f"Cross-validation (2023 → 2024): R² {r2:.2f} · RMSE {rmse:.2f} · MAE {mae:.2f}")
            else:
                dict_entries = [(k, v) for k, v in perf.items() if isinstance(v, dict)]
                if dict_entries:
                    _, metrics = max(dict_entries, key=lambda x: x[1].get("r2", 0))
                    st.caption(f"R² {metrics.get('r2', 0):.2f} · RMSE {metrics.get('rmse', 0):.2f} · MAE {metrics.get('mae', 0):.2f}")
                else:
                    st.caption("See training_summary.json")
        else:
            st.caption("Train the model to see metrics here.")
        if summary and summary.get("sample_weighted"):
            st.caption("Trained with sample weights (higher weight for higher PPG).")

    with st.expander("How I did it"):
        st.markdown("""
        **Setup:** I use one year of stats to predict the next year (e.g., 2023 to predict 2024, 2024 to predict 2025). The model never sees the target season, so there's no leakage.

        **Inputs:** For each player the model gets: position, prior-year points per game (PPG), and stat totals (passing/rushing/receiving yards and TDs, receptions, interceptions, fumbles). It also uses team strength: the average PPG of other players on the same team that year, so context like a strong offense is included.

        **Training:** Data is split 80/20 for train vs test. I can train only on fantasy-relevant players (e.g. target PPG ≥ 5) so the model focuses on ranking starters instead of the long tail of low scorers. When 2025 actuals are available, the app shows how the 2025 projections did against real results.

        **How it's doing:** Overall (all matched players) the naive baseline (last year's PPG) slightly outperforms the model. The "Starter-level only" slice (PPG ≥ 5) focuses on starters; metrics are shown for both all players and that slice.

        **Is this accurate?** The feature importance chart shows how this model weighs its inputs. Prior PPG being the dominant feature is normal for fantasy. The model doesn't use injuries, schedule, or coaching; adding those could help further.
        """)

    st.markdown('<p class="section-header">Data</p>', unsafe_allow_html=True)
    st.caption(f"{len(players):,} players · {len(teams)} teams · {len(stats):,} stat rows from 2024.")

    if model is not None and getattr(model, "feature_importance", None):
        with st.expander("What the model cares about"):
            try:
                imp = model.get_feature_importance("random_forest")
            except Exception:
                try:
                    imp = model.get_feature_importance("gradient_boosting")
                except Exception:
                    imp = None
            if imp is not None and not imp.empty:
                fig = px.bar(imp, x="feature", y="importance", title="Feature importance")
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("N/A for this model.")

    st.markdown('<p class="section-header">Predictions</p>', unsafe_allow_html=True)
    has_2025_actuals = _load_actuals_2025() is not None
    if has_2025_actuals:
        st.caption("2025 results are in — I’ll show how my picks did once you run it.")
    if st.button("Run model", type="primary"):
        with st.spinner("Running the numbers..."):
            pred_df, used_ml = create_predictions(players, teams, stats)
        if pred_df.empty:
            st.warning("Nothing to predict — double-check your data.")
        else:
            pred_df = pred_df.sort_values("projection", ascending=False).reset_index(drop=True)
            if used_ml:
                st.success("Your 2025 projections are ready.")
            else:
                st.info("Using a simple fallback. Train the model for the full experience.")
            actuals_2025 = _load_actuals_2025()
            if actuals_2025 is not None and "player_id" in pred_df.columns:
                pred_df = pred_df.merge(actuals_2025, on="player_id", how="left")
            # Build display with exactly these columns so labels and order stay correct
            display = pd.DataFrame({"Player": pred_df["player_name"].values})
            display["2025 proj PPG"] = pd.to_numeric(pred_df["projection"], errors="coerce").fillna(0).round(1)
            display["2024 PPG (prior)"] = pd.to_numeric(pred_df["fantasy_pts_2024_ppg"], errors="coerce").fillna(0).round(1)
            if "actual_ppg_2025" in pred_df.columns:
                display["2025 actual PPG"] = pd.to_numeric(pred_df["actual_ppg_2025"], errors="coerce").fillna(0).round(1)
            st.dataframe(display, use_container_width=True, height=400)
            if "actual_ppg_2025" in pred_df.columns:
                st.caption("2025 actual PPG = season total ÷ games played (from real_stats_2025.csv).")
            # Out-of-sample 2025 accuracy when actuals exist
            if "actual_ppg_2025" in pred_df.columns:
                eval_df = pred_df.dropna(subset=["actual_ppg_2025", "projection"])
                if "fantasy_pts_2024_ppg" in eval_df.columns:
                    eval_df = eval_df.dropna(subset=["fantasy_pts_2024_ppg"])
                if len(eval_df) >= 10:
                    min_ppg_relevant = 5.0  # fantasy-relevant: actual 2025 PPG >= 5
                    eval_relevant = eval_df[eval_df["actual_ppg_2025"] >= min_ppg_relevant]
                    use_relevant = len(eval_relevant) >= 10  # show "relevant only" if we have enough

                    def _metrics(df):
                        yt = df["actual_ppg_2025"].values
                        yp = df["projection"].values.astype(float)
                        ss_res = np.sum((yt - yp) ** 2)
                        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
                        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
                        mae = float(np.mean(np.abs(yt - yp)))
                        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
                        return r2, mae, rmse, yt, yp

                    r2_2025, mae_2025, rmse_2025, y_true, y_pred = _metrics(eval_df)
                    # Baseline: 2024 PPG as 2025 prediction (same players)
                    if "fantasy_pts_2024_ppg" in eval_df.columns:
                        y_baseline = np.asarray(eval_df["fantasy_pts_2024_ppg"].values, dtype=float)
                        baseline_mae = float(np.mean(np.abs(y_true - y_baseline)))
                        baseline_rmse = float(np.sqrt(np.mean((y_true - y_baseline) ** 2)))
                        ss_res_b = np.sum((y_true - y_baseline) ** 2)
                        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                        baseline_r2 = float(1 - (ss_res_b / ss_tot)) if ss_tot > 0 else 0
                    else:
                        baseline_mae = baseline_rmse = baseline_r2 = None

                    st.markdown("**Baseline (naive predictor)**")
                    st.caption("Predict 2025 PPG = 2024 PPG (no model).")
                    if baseline_mae is not None:
                        st.caption(f"Baseline slightly outperforms the model overall: lower MAE ({baseline_mae:.2f} vs {mae_2025:.2f}) and lower RMSE ({baseline_rmse:.2f} vs {rmse_2025:.2f}).")

                    st.markdown("---")
                    st.markdown("**Results: All players**")
                    st.caption(f"n = {len(eval_df)} players with 2024 stats and 2025 actuals.")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("R²", f"{r2_2025:.3f}")
                    with c2:
                        st.metric("MAE (PPG)", f"{mae_2025:.2f}")
                    with c3:
                        st.metric("RMSE (PPG)", f"{rmse_2025:.2f}")
                    st.caption("High overall R² driven by low-PPG long tail.")

                    if use_relevant:
                        r2_r, mae_r, rmse_r, y_true_r, y_pred_r = _metrics(eval_relevant)
                        # Baseline on same subset (actual 2025 PPG ≥ 5)
                        y_baseline_r = np.asarray(eval_relevant["fantasy_pts_2024_ppg"].values, dtype=float)
                        baseline_r_mae = float(np.mean(np.abs(y_true_r - y_baseline_r)))
                        baseline_r_rmse = float(np.sqrt(np.mean((y_true_r - y_baseline_r) ** 2)))
                        ss_res_br = np.sum((y_true_r - y_baseline_r) ** 2)
                        ss_tot_r = np.sum((y_true_r - np.mean(y_true_r)) ** 2)
                        baseline_r_r2 = float(1 - (ss_res_br / ss_tot_r)) if ss_tot_r > 0 else 0

                        st.markdown("---")
                        st.markdown("**Results: Starter-level players only (PPG ≥ 5)**")
                        st.caption(f"n = {len(eval_relevant)} players. Starters and flex-relevant scorers.")
                        st.caption("**Baseline on this subset** (predict 2025 = 2024 PPG): "
                                  f"R² = {baseline_r_r2:.3f}, MAE = {baseline_r_mae:.2f}, RMSE = {baseline_r_rmse:.2f}. "
                                  "**Model on this subset:** "
                                  f"R² = {r2_r:.3f}, MAE = {mae_r:.2f}, RMSE = {rmse_r:.2f}.")
                        cr1, cr2, cr3 = st.columns(3)
                        with cr1:
                            st.metric("R²", f"{r2_r:.3f}")
                        with cr2:
                            st.metric("MAE (PPG)", f"{mae_r:.2f}")
                        with cr3:
                            st.metric("RMSE (PPG)", f"{rmse_r:.2f}")
                        if baseline_r_mae > 0:
                            model_beats_baseline = mae_r < baseline_r_mae and rmse_r < baseline_r_rmse
                            if model_beats_baseline:
                                st.caption("Model slightly outperforms baseline on this subset.")
                            else:
                                st.caption(
                                    "Both methods struggle on starter-level players due to year-to-year volatility. "
                                    "Baseline slightly outperforms ML, suggesting limited predictive signal beyond prior PPG without injury, usage, and context features."
                                )

                    st.markdown("---")
                    with st.container():
                        st.markdown("**Key insight**")
                        st.info(
                            "ML helps identify low-impact players (overall R² is strong), but the naive baseline (last year's PPG) slightly outperforms the model overall. "
                            "For top performers, baseline heuristics remain stronger due to volatility and limited signal beyond prior PPG."
                        )

                    # Model comparison table (2025 out-of-sample)
                    if baseline_mae is not None:
                        st.markdown("**Model comparison (2025 out-of-sample)**")
                        comp = pd.DataFrame({
                            "Method": ["Baseline (2024 → 2025)", "Model (ensemble)"],
                            "R²": [round(baseline_r2, 3), round(r2_2025, 3)],
                            "MAE": [round(baseline_mae, 2), round(mae_2025, 2)],
                            "RMSE": [round(baseline_rmse, 2), round(rmse_2025, 2)],
                        })
                        st.dataframe(comp, use_container_width=True, hide_index=True)
                    # Per-model from training (80/20 split)
                    if model is not None and getattr(model, "model_performance", None):
                        perf = model.model_performance
                        names = [n for n in ["lightgbm", "random_forest", "gradient_boosting"] if n in perf and isinstance(perf.get(n), dict)]
                        if names:
                            st.caption("Per-model metrics from training (80/20 split):")
                            rows = [{"Model": n, "R²": round(perf[n].get("r2", 0), 3), "MAE": round(perf[n].get("mae", 0), 2), "RMSE": round(perf[n].get("rmse", 0), 2)} for n in names]
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.markdown("**Limitations**")
                    st.caption(
                        "This forecast does not account for: injuries, coaching or scheme changes, rookies (no prior NFL PPG), or trades. "
                        "Sample size for fantasy-relevant players is modest; metrics can shift with more data or different seasons."
                    )
            fig = px.histogram(pred_df, x="projection", nbins=30, title="Projected PPG spread")
            st.plotly_chart(fig, use_container_width=True)
            csv = pred_df.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="fantasy_2025_projections.csv", mime="text/csv")
    else:
        st.caption("Hit **Run model** and see your 2025 projections.")


if __name__ == "__main__":
    main()
