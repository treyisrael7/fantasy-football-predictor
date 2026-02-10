# 🏈 Fantasy Football Predictor

A production-ready AI/ML system for predicting NFL fantasy football performance using real data from the Sleeper API.

## ✨ Features

- **Real NFL Data**: 2,400+ real players from Sleeper API
- **Live Predictions**: Real-time 2025 season data
- **ML Models**: Trained on 2024 historical data (R² = 0.902)
- **Web Interface**: Beautiful Streamlit dashboard
- **Production Ready**: Docker, PostgreSQL, Redis, monitoring

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Real Data
```bash
python scripts/data_collection.py
```

### 3. Train Models
```bash
python scripts/train_models.py
```

### 4. Run Web App
```bash
streamlit run app/main.py
```

## 📊 Data Sources

- **Sleeper API**: Real NFL players, teams, and statistics
- **Live Data**: Current 2025 season performance
- **Historical**: 2024 season data for model training

## 🎯 Model Performance

- **General Model**: R² = 0.902, RMSE = 1.595
- **Position Models**: QB, RB, WR, TE, K, DEF specific models
- **Real Players**: Lamar Jackson, Ja'Marr Chase, CeeDee Lamb, etc.

## 🏗️ Architecture

```
├── app/                 # Web application
├── src/                 # Source code
│   ├── data_collection/ # Real data collection
│   ├── models/         # ML models
│   ├── preprocessing/  # Feature engineering
│   └── utils/          # Utilities
├── scripts/            # Data collection & training
├── data/               # Real NFL data
├── models/             # Trained ML models
└── docs/               # Documentation
```

## 🛠️ Production Deployment

### Deploy on a single server (portfolio)

**What you do outside the codebase:**

1. **Install Docker and Docker Compose** on the server (e.g. Ubuntu: `sudo apt update && sudo apt install -y docker.io docker-compose`; then `sudo usermod -aG docker $USER` and log out/in).

2. **Clone the repo** and `cd` into the project directory.

3. **Ensure data and models exist** (the app needs them on first run):
   - Either copy your existing `data/` and `models/` folders onto the server into the project root, **or**
   - Run once without Docker to generate them:
     ```bash
     pip install -r requirements.txt
     python scripts/data_collection.py
     python scripts/train_models.py
     ```
   - Then start with Docker (step 4). The app will **auto-seed** Postgres from `data/*.csv` the first time it runs.

4. **Start the stack:**
   ```bash
   docker-compose up -d
   ```
   This starts the **web app** (Streamlit), **PostgreSQL**, and **Redis**. The web app reads data from Redis → Postgres → CSV and seeds Postgres from CSV if the DB is empty.

5. **Optional:** To also run the FastAPI backend (e.g. for a separate API URL):
   ```bash
   docker-compose --profile api up -d
   ```

6. **Optional:** Override secrets for production: copy `.env.example` to `.env`, set `DATABASE_URL` and `REDIS_URL` if not using defaults, and use stronger passwords. Docker Compose reads `.env` automatically.

**No API keys** are required for Sleeper; data is loaded from your existing CSVs and stored in Postgres/Redis.

### Docker (Recommended)
```bash
# Start all services (web + Postgres + Redis)
docker-compose up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# Start with API
docker-compose --profile api up -d
```

### Services
- **Web App**: http://localhost:8501
- **API**: http://localhost:8000 (when run with `--profile api`)
- **Database**: PostgreSQL (port 5432) — app uses it when `USE_DB=true`
- **Cache**: Redis (port 6379) — caches players/teams/stats
- **Monitoring**: Prometheus (port 9090) + Grafana (port 3000)

### Docker Commands
```bash
# Build and start
docker-compose up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v
```

## 📈 Live Data

The system automatically collects live 2025 NFL data:
- **Players**: 7,670+ real NFL players
- **Stats**: Current week performance
- **Predictions**: Real-time fantasy projections

## 🎯 Sample Predictions

| Player | Position | Team | Projection | Confidence |
|--------|----------|------|------------|------------|
| CeeDee Lamb | WR | DAL | 1.3 | 87.9% |
| Kyle Pitts | TE | ATL | 0.3 | 64.0% |
| Joe Flacco | QB | CLE | 0.3 | 73.0% |

## 🔧 Configuration

- **Data Source**: Sleeper API (free, no API key required)
- **Update Frequency**: Every 6 hours + daily/weekly
- **Model Retraining**: Manual (can be automated)

## 📝 License

MIT License - feel free to use for your own projects!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for fantasy football enthusiasts**
