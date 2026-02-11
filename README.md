# Fantasy Football Forecasting Dashboard

I built this to see how well a simple ML setup could predict next-year fantasy PPG compared to the obvious baseline: just using last year’s PPG. It uses prior-season stats (and optional team/roster context), trains on 2023→2024, and evaluates on 2025 actuals when you have them. All the metrics—baseline vs model, all players vs starter-level only—live in the app so you can see where the model helps and where it doesn’t.

## What you’ll see in the app

The dashboard shows cross-validation metrics (2023→2024) and, when 2025 actuals are available, out-of-sample results. You get baseline (predict 2025 = 2024 PPG) vs the ML ensemble, and the same comparison for starter-level players only (PPG ≥ 5). Honest take: the naive baseline often does about as well or better overall; the model is still useful for context and for seeing where extra features don’t buy much.

**Live app:** [https://fantasy-football-predictor.streamlit.app/](https://fantasy-football-predictor.streamlit.app/)

---
