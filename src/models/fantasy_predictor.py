"""
Fantasy Football Predictor Models
Implements various ML models for predicting fantasy football performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FantasyPredictor:
    """Main class for fantasy football predictions using ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.is_trained = False
        
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train multiple models for ensemble prediction
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            random_state: Random seed
        
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training fantasy football prediction models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler
        
        # Define models
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        # Train models
        performance = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            try:
                # Use scaled data for linear models, original for tree-based
                if name in ['linear_regression', 'ridge', 'lasso']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                performance[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Store model
                self.models[name] = model
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(
                        zip(X.columns, model.feature_importances_)
                    )
                
                logger.info(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        self.model_performance = performance
        self.is_trained = True
        
        return performance
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using trained models
        
        Args:
            X: Feature matrix
            model_name: Specific model to use (if None, uses ensemble)
        
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            if model_name in ['linear_regression', 'ridge', 'lasso']:
                X_scaled = self.scalers['main'].transform(X)
                return model.predict(X_scaled)
            else:
                return model.predict(X)
        else:
            # Ensemble prediction (weighted average)
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                if name in ['linear_regression', 'ridge', 'lasso']:
                    X_scaled = self.scalers['main'].transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                predictions.append(pred)
                # Weight by R² score
                weight = self.model_performance[name]['r2']
                weights.append(weight)
            
            # Weighted average
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return ensemble_pred
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        predictions = []
        
        for name, model in self.models.items():
            if name in ['linear_regression', 'ridge', 'lasso']:
                X_scaled = self.scalers['main'].transform(X)
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)
        
        # Confidence based on prediction variance
        pred_std = np.std(predictions, axis=0)
        confidence = 1 / (1 + pred_std)  # Higher variance = lower confidence
        
        return mean_pred, confidence
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance from a specific model
        
        Args:
            model_name: Name of the model
        
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.feature_importance:
            raise ValueError(f"Feature importance not available for {model_name}")
        
        importance_df = pd.DataFrame(
            list(self.feature_importance[model_name].items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.model_performance = model_data['model_performance']
        self.is_trained = model_data['is_trained']
        logger.info(f"Models loaded from {filepath}")

class PositionSpecificPredictor:
    """Position-specific predictors for different player types"""
    
    def __init__(self):
        self.position_models = {}
        self.position_features = {
            'QB': ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds'],
            'RB': ['rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds', 'carries', 'receptions'],
            'WR': ['receiving_yards', 'receiving_tds', 'receptions', 'targets', 'rushing_yards'],
            'TE': ['receiving_yards', 'receiving_tds', 'receptions', 'targets'],
            'K': ['field_goals_made', 'field_goals_attempted', 'extra_points_made'],
            'DEF': ['points_allowed', 'sacks', 'interceptions', 'fumble_recoveries', 'defensive_tds']
        }
    
    def train_position_models(self, df: pd.DataFrame, target_col: str = 'target_fantasy_points'):
        """Train separate models for each position"""
        logger.info("Training position-specific models")
        
        for position in self.position_features.keys():
            logger.info(f"Training {position} model")
            
            # Filter data for position
            pos_data = df[df['position'] == position].copy()
            
            if len(pos_data) < 50:  # Need minimum data
                logger.warning(f"Insufficient data for {position} model")
                continue
            
            # Select features
            feature_cols = [col for col in self.position_features[position] if col in pos_data.columns]
            
            # Add engineered features
            engineered_features = [
                'fantasy_points_last_3_avg', 'fantasy_points_last_5_avg',
                'fantasy_trend', 'opp_def_strength', 'is_home',
                'cold_weather', 'windy_conditions', 'injury_multiplier'
            ]
            
            feature_cols.extend([col for col in engineered_features if col in pos_data.columns])
            
            X = pos_data[feature_cols]
            y = pos_data[target_col]
            
            # Train model
            predictor = FantasyPredictor()
            performance = predictor.train_models(X, y)
            
            self.position_models[position] = {
                'predictor': predictor,
                'features': feature_cols,
                'performance': performance
            }
            
            logger.info(f"{position} model trained - Best R²: {max([p['r2'] for p in performance.values()]):.3f}")
    
    def predict_position(self, X: pd.DataFrame, position: str) -> np.ndarray:
        """Make predictions for a specific position"""
        if position not in self.position_models:
            raise ValueError(f"No model trained for position {position}")
        
        model_info = self.position_models[position]
        predictor = model_info['predictor']
        features = model_info['features']
        
        # Select relevant features
        X_filtered = X[[col for col in features if col in X.columns]]
        
        return predictor.predict(X_filtered)

# Example usage
if __name__ == "__main__":
    from data_collection.sleeper_collector import SleeperNFLCollector
    from preprocessing.feature_engineering import FeatureEngineer
    
    collector = SleeperNFLCollector()
    engineer = FeatureEngineer()
    
    all_data = []
    for week in range(1, 6):
        week_data = collector.get_player_stats(2024, week)
        all_data.append(week_data)
    
    df = pd.concat(all_data, ignore_index=True)
    
    # Engineer features
    df = engineer.create_player_features(df)
    df = engineer.create_contextual_features(df)
    df = engineer.create_target_variable(df)
    
    # Prepare features
    feature_cols = engineer.get_feature_columns()
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].fillna(0)
    y = df['target_fantasy_points']
    
    # Train models
    predictor = FantasyPredictor()
    performance = predictor.train_models(X, y)
    
    print("Model Performance:")
    for model, metrics in performance.items():
        print(f"{model}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}")
    
    # Make predictions
    predictions = predictor.predict(X.head(10))
    print(f"\nSample predictions: {predictions[:5]}")
    
    # Position-specific training
    pos_predictor = PositionSpecificPredictor()
    pos_predictor.train_position_models(df)
    
    print(f"\nPosition models trained: {list(pos_predictor.position_models.keys())}")
