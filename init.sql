-- Fantasy Football Predictor Database Schema
-- Runs in POSTGRES_DB (fantasy_football) when using Docker.

-- Create tables
CREATE TABLE IF NOT EXISTS players (
    player_id VARCHAR(50) PRIMARY KEY,
    player_name VARCHAR(100) NOT NULL,
    position VARCHAR(10) NOT NULL,
    team VARCHAR(10) NOT NULL,
    status VARCHAR(20) DEFAULT 'Active',
    height INTEGER,
    weight INTEGER,
    college VARCHAR(100),
    years_exp DECIMAL(3,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teams (
    team_id VARCHAR(10) PRIMARY KEY,
    team VARCHAR(10) NOT NULL,
    name VARCHAR(100) NOT NULL,
    conference VARCHAR(10) NOT NULL,
    division VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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
);

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
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    r2_score DECIMAL(6,4),
    mae DECIMAL(6,4),
    rmse DECIMAL(6,4),
    training_samples INTEGER,
    test_samples INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team);
CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);
CREATE INDEX IF NOT EXISTS idx_stats_player_season ON player_stats(player_id, season);
CREATE INDEX IF NOT EXISTS idx_predictions_player_week ON predictions(player_id, week, season);

-- Insert sample data (optional)
INSERT INTO teams (team_id, team, name, conference, division) VALUES
('ARI', 'ARI', 'Arizona Cardinals', 'NFC', 'West'),
('ATL', 'ATL', 'Atlanta Falcons', 'NFC', 'South'),
('BAL', 'BAL', 'Baltimore Ravens', 'AFC', 'North'),
('BUF', 'BUF', 'Buffalo Bills', 'AFC', 'East'),
('CAR', 'CAR', 'Carolina Panthers', 'NFC', 'South'),
('CHI', 'CHI', 'Chicago Bears', 'NFC', 'North'),
('CIN', 'CIN', 'Cincinnati Bengals', 'AFC', 'North'),
('CLE', 'CLE', 'Cleveland Browns', 'AFC', 'North'),
('DAL', 'DAL', 'Dallas Cowboys', 'NFC', 'East'),
('DEN', 'DEN', 'Denver Broncos', 'AFC', 'West'),
('DET', 'DET', 'Detroit Lions', 'NFC', 'North'),
('GB', 'GB', 'Green Bay Packers', 'NFC', 'North'),
('HOU', 'HOU', 'Houston Texans', 'AFC', 'South'),
('IND', 'IND', 'Indianapolis Colts', 'AFC', 'South'),
('JAX', 'JAX', 'Jacksonville Jaguars', 'AFC', 'South'),
('KC', 'KC', 'Kansas City Chiefs', 'AFC', 'West'),
('LV', 'LV', 'Las Vegas Raiders', 'AFC', 'West'),
('LAC', 'LAC', 'Los Angeles Chargers', 'AFC', 'West'),
('LAR', 'LAR', 'Los Angeles Rams', 'NFC', 'West'),
('MIA', 'MIA', 'Miami Dolphins', 'AFC', 'East'),
('MIN', 'MIN', 'Minnesota Vikings', 'NFC', 'North'),
('NE', 'NE', 'New England Patriots', 'AFC', 'East'),
('NO', 'NO', 'New Orleans Saints', 'NFC', 'South'),
('NYG', 'NYG', 'New York Giants', 'NFC', 'East'),
('NYJ', 'NYJ', 'New York Jets', 'AFC', 'East'),
('PHI', 'PHI', 'Philadelphia Eagles', 'NFC', 'East'),
('PIT', 'PIT', 'Pittsburgh Steelers', 'AFC', 'North'),
('SF', 'SF', 'San Francisco 49ers', 'NFC', 'West'),
('SEA', 'SEA', 'Seattle Seahawks', 'NFC', 'West'),
('TB', 'TB', 'Tampa Bay Buccaneers', 'NFC', 'South'),
('TEN', 'TEN', 'Tennessee Titans', 'AFC', 'South'),
('WAS', 'WAS', 'Washington Commanders', 'NFC', 'East')
ON CONFLICT (team_id) DO NOTHING;
