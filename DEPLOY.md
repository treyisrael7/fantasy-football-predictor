# How to run the app

## Local

1. `cd` into the project folder.
2. Install deps, get data, and train (see README Quick Start).
3. Run:
   ```bash
   streamlit run app/main.py
   ```
4. Open **http://localhost:8501** in your browser.

---

## Public deploy (Streamlit Community Cloud)

1. **Push your repo to GitHub** including `data/` (real_players.csv, real_teams.csv, real_stats_2024.csv, etc.) and, if you have them, trained models in `models/`. The app needs these to run in the cloud.

2. **Include data so the app can load:**
   - By default `data/*.csv` is in `.gitignore`, so the app will show "Couldn't find data" on first run.
   - **Option A:** Generate CSVs locally, then **temporarily** allow them in git so the cloud app has data:
     - Comment out or remove the `data/*.csv` line from `.gitignore`, then:
     - `git add data/*.csv` (and `models/` if you want trained models), commit, and push.
   - **Option B:** Keep `data/*.csv` ignored and add a **small sample** (e.g. a few hundred rows) in a folder you do commit, and point the app at it via an env var—only if you’re comfortable changing the app’s `data_dir`. Easiest for a portfolio is Option A with a small subset of CSVs committed.

3. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - **New app** → choose this repo and branch.
   - **Main file path:** `app/main.py`
   - **Advanced settings** → Python version: 3.10 or 3.11 (match your local if possible).
   - Deploy. You’ll get a link like `https://your-app-name.streamlit.app`.

3. **Limits:** Free apps may sleep after ~30 min of no use; the next visitor wakes it (short delay). No credit card required.
