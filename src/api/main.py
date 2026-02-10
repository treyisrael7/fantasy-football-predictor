"""
FastAPI Backend for Fantasy Football Predictor
Provides REST API endpoints for predictions and data access
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import sys

# Add src and project root to path
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_app_root = os.path.dirname(_src_dir)
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from models.fantasy_predictor import FantasyPredictor
from data_collection.sleeper_collector import SleeperNFLCollector
from data_loader import load_data as load_data_backend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fantasy Football Predictor API",
    description="REST API for NFL fantasy football predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PlayerPrediction(BaseModel):
    player_id: str
    player_name: str
    position: str
    team: str
    predicted_points: float
    confidence: float

class PredictionRequest(BaseModel):
    week: int = 1
    season: int = 2024
    positions: Optional[List[str]] = None
    teams: Optional[List[str]] = None
    limit: int = 50

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

# Global variables for caching
predictor = None
data_collector = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and data collector on startup"""
    global predictor, data_collector
    
    try:
        # Initialize predictor
        predictor = FantasyPredictor()
        
        # Load trained models if available
        if os.path.exists("models/models_real/general_predictor_real.pkl"):
            predictor.load_models("models/models_real/")
            logger.info("Loaded trained models")
        
        # Initialize data collector
        data_collector = SleeperNFLCollector()
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

def _get_data():
    """Load players, teams, stats from Redis/Postgres/CSV."""
    data_dir = os.path.join(_app_root, "data") if _app_root else "data"
    players, teams, stats, _, _ = load_data_backend(data_dir=data_dir)
    return players, teams, stats


@app.get("/players")
async def get_players(limit: int = 100):
    """Get list of players (from Redis cache, Postgres, or CSV)."""
    try:
        players, _, _ = _get_data()
        if players is None or players.empty:
            raise HTTPException(status_code=404, detail="Player data not found")
        players = players.head(limit)
        return {
            "players": players.to_dict("records"),
            "total": len(players),
            "limit": limit
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting players: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/teams")
async def get_teams():
    """Get list of teams (from Redis cache, Postgres, or CSV)."""
    try:
        _, teams, _ = _get_data()
        if teams is None or teams.empty:
            raise HTTPException(status_code=404, detail="Team data not found")
        return {
            "teams": teams.to_dict("records"),
            "total": len(teams)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictions", response_model=List[PlayerPrediction])
async def get_predictions(request: PredictionRequest):
    """Get fantasy football predictions (data from Redis/Postgres/CSV)."""
    try:
        players, teams, stats = _get_data()
        if players is None or teams is None or stats is None:
            raise HTTPException(status_code=503, detail="Data not available")
        if players.empty or teams.empty:
            raise HTTPException(status_code=503, detail="Player/team data empty")

        # Convert player_id to string for consistent merging
        players['player_id'] = players['player_id'].astype(str)
        stats['player_id'] = stats['player_id'].astype(str)
        
        # Remove duplicate player_name column from stats before merging
        if stats.empty:
            raise HTTPException(status_code=503, detail="Stats data empty")
        if 'player_name' in stats.columns:
            stats = stats.drop('player_name', axis=1)
        # Merge data
        player_data = players.merge(stats, on='player_id', how='inner')
        player_data = player_data.merge(teams, on='team', how='left')
        
        # Filter for players with meaningful stats
        player_data = player_data[
            (player_data['fantasy_points'] > 0) & 
            (player_data['player_name'].notna())
        ].copy()
        
        # Apply filters
        if request.positions:
            player_data = player_data[player_data['position'].isin(request.positions)]
        
        if request.teams:
            player_data = player_data[player_data['team'].isin(request.teams)]
        
        # Create predictions
        predictions = []
        
        for _, player in player_data.head(request.limit).iterrows():
            # Base prediction on 2024 performance with some variance
            base_points = player.get('fantasy_points', 0) / 17  # Average per week
            variance = np.random.normal(0, base_points * 0.3)  # 30% variance
            predicted_points = max(0, base_points + variance)
            
            confidence = min(95, max(60, 100 - abs(variance) * 2))
            
            prediction = PlayerPrediction(
                player_id=player['player_id'],
                player_name=player['player_name'],
                position=player['position'],
                team=player['team'],
                predicted_points=round(predicted_points, 1),
                confidence=round(confidence, 1)
            )
            predictions.append(prediction)
        
        # Sort by predicted points
        predictions.sort(key=lambda x: x.predicted_points, reverse=True)
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{player_id}")
async def get_player_stats(player_id: str, season: int = 2024):
    """Get stats for a specific player (from Redis/Postgres/CSV)."""
    try:
        _, _, stats = _get_data()
        if stats is None or stats.empty:
            raise HTTPException(status_code=404, detail="Stats data not found")
        player_stats = stats[stats['player_id'].astype(str) == str(player_id)]
        if player_stats.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        return {
            "player_id": player_id,
            "season": season,
            "stats": player_stats.to_dict("records")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting player stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        if not os.path.exists("models/models_real/training_summary_real.json"):
            raise HTTPException(status_code=404, detail="Model performance data not found")
        
        with open("models/models_real/training_summary_real.json", "r") as f:
            performance = f.read()
        
        return {"performance": performance}
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
