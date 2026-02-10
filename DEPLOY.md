# Deploy steps (what you do in the terminal)

Everything below is something **you run in a terminal** (Command Prompt, PowerShell, or a terminal on a Linux server). You’re not editing code for these steps—just running commands.

**Where to run:** Open a terminal and `cd` into your project folder so you’re inside the Fantasy Football Predictor folder (where `docker-compose.yml` and `app/` live). All commands assume you’re in that folder.

---

## Option A: Run on your own computer (local)

Use this to try the app with Postgres and Redis on your machine.

1. **Install Docker Desktop**  
   - Download and install from [docker.com](https://www.docker.com/products/docker-desktop/).  
   - You do this once; not in the project folder.

2. **Open a terminal in the project folder**  
   - In VS Code / Cursor: Terminal → New Terminal (it usually opens in the project folder).  
   - Or: `cd` to the folder, e.g.  
     `cd "c:\Users\Owner\OneDrive - g.clemson.edu\Desktop\Fantasy Football Predictor"`

3. **Make sure data and models exist**  
   You already have `data/real_players.csv`, etc. If you also have the trained models in `models/models_real/`, you’re set.  
   If you ever need to regenerate data or models (from the project folder):

   ```bash
   pip install -r requirements.txt
   python scripts/data_collection.py
   python scripts/train_models.py
   ```

4. **Start the app with Docker (in the same terminal, same folder)**  

   ```bash
   docker-compose up -d
   ```

   If the build fails with an SSL error (e.g. on campus or corporate networks), try:
   ```bash
   docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1
   docker-compose up -d
   ```
   On a normal cloud server, the regular `docker-compose up -d` usually works.

   That’s it. The app will start and seed Postgres from your CSV the first time.

5. **Open the app**  
   In your browser go to: **http://localhost:8501**

6. **Stop everything when you’re done**  

   ```bash
   docker-compose down
   ```

---

## Option B: Put it on a server (e.g. for your portfolio)

Use this when you want the app to run on a VPS/cloud server so others can visit it.

1. **On the server (SSH into it, then in the terminal)**  
   - Install Docker (example for Ubuntu):  
     `sudo apt update && sudo apt install -y docker.io docker-compose`  
   - Add your user to the docker group and log out and back in:  
     `sudo usermod -aG docker $USER`

2. **Get the project onto the server**  
   - Either clone the repo:  
     `git clone <your-repo-url>`  
     `cd Fantasy-Football-Predictor`  
   - Or copy the whole project folder (including `data/` and `models/`) onto the server and `cd` into that folder in the terminal.

3. **In the terminal, in the project folder on the server**  
   - If you didn’t copy `data/` and `models/`, generate them once:  
     `pip install -r requirements.txt`  
     `python scripts/data_collection.py`  
     `python scripts/train_models.py`  
   - Start the app:  
     `docker-compose up -d`  
     (If the build ever fails with an SSL error on the server, use  
     `docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1` then `docker-compose up -d`.)

4. **Open the app**  
   In a browser, go to: **http://&lt;your-server-ip&gt;:8501**  
   (Replace &lt;your-server-ip&gt; with the server’s public IP or domain.)

---

## Option C: Free public link (like Vercel) — so anyone can open it

You want one URL you can share (e.g. `https://something.whatever.com`), for free.

**Vercel won’t work** for this project. Vercel is for static sites and serverless; this app is a long‑running Streamlit server with Postgres and Redis. Use one of these free options instead.

---

### C1. Streamlit Community Cloud (easiest, 100% free)

You get a link like **`https://your-app-name.streamlit.app`**. No credit card, no Docker on your side.

1. **Push your project to GitHub** (including the `data/` folder with `real_players.csv`, `real_teams.csv`, `real_stats_2024.csv`, and `models/models_real/` if you have them). If `data/` or `models/` are in `.gitignore`, you’ll need to either commit them or use a free DB (step 2).

2. **Create a free Postgres database** (so the app has somewhere to store/read data):
   - **Supabase (recommended if you’ve used it before):** [supabase.com](https://supabase.com) → New project → Settings → Database → copy the **URI** (connection string). Use “Transaction” mode; it looks like `postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres`.
   - **Or Neon:** [neon.tech](https://neon.tech) → sign up → create a project → copy the connection string (e.g. `postgresql://user:pass@ep-xxx.us-east-1.aws.neon.tech/neondb?sslmode=require`).

3. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **“New app”** → pick your repo and branch.
   - **Main file path:** `app/main.py`
   - **Advanced settings** → “Secrets”: add:
     ```toml
     DATABASE_URL = "postgresql://..."   # paste your Neon connection string
     USE_DB = "true"
     ```
   - (Optional) For Redis caching, create a free DB at [upstash.com](https://upstash.com) (Redis), copy the URL, and add:
     ```toml
     REDIS_URL = "rediss://..."
     ```
   - Deploy. After a few minutes you get a public link.

4. **First load:** The app will seed Postgres from the CSV files in your repo. Your `data/real_players.csv`, `data/real_teams.csv`, and `data/real_stats_2024.csv` are not in `.gitignore`, so they’re already committed—the app will use them. Trained `.pkl` models are gitignored; the app still runs and shows predictions without them.

**Limits:** Free apps may sleep after ~30 min of no use; the next visitor wakes it (short delay). Fine for a portfolio.

---

### C2. Render.com (Docker + free Postgres + Redis)

You get a link like **`https://your-app-name.onrender.com`**. Uses your existing Dockerfile; Render runs it and gives you free Postgres and Redis.

1. **Push your project to GitHub** (include `data/` and `models/` or the app will need to seed from CSV; Render will build from the Dockerfile).

2. **Sign up at [render.com](https://render.com)** (free).

3. **Create a free Postgres database:** Dashboard → New → PostgreSQL. Copy the **Internal Database URL** (use this inside Render; it’s only for your app).

4. **Create a free Redis:** Dashboard → New → Redis. Copy the **Internal Redis URL**.

5. **Create a Web Service:** Dashboard → New → Web Service → connect your GitHub repo.
   - **Environment:** Docker.
   - **Build command:** leave default (Render uses your Dockerfile).
   - **Start command:** leave default (your Dockerfile CMD runs Streamlit).
   - **Environment variables:** Add:
     - `DATABASE_URL` = (paste the Postgres Internal URL from step 3)
     - `REDIS_URL` = (paste the Redis Internal URL from step 4)
   - You can use Supabase or Neon for Postgres instead of Render’s DB if you prefer; then set `DATABASE_URL` to that connection string.
   - Create Web Service. Render builds and deploys; you get a public URL.

6. **Important:** The app runs in one container; it connects to Render’s Postgres and Redis (different containers). So your app will seed Postgres from CSV on first run if the DB is empty. Ensure `data/` is in the repo so the CSV files are in the image.

**Limits:** Free tier spins down after ~15 min of no traffic; first visit after that has a cold start (30–60 seconds). Still free and shareable.

---

## Quick reference

| What you want              | Where you run it     | What to do |
|----------------------------|----------------------|------------|
| Run app locally            | Your PC terminal     | `docker-compose up -d` |
| Stop the app               | Same terminal        | `docker-compose down` |
| Regenerate data             | Same folder          | `python scripts/data_collection.py` |
| Regenerate models           | Same folder          | `python scripts/train_models.py` |
| Deploy on a server (VPS)    | Server terminal (SSH)| `docker-compose up -d` in project folder |
| **Free public link (share with everyone)** | Browser + GitHub | **Option C:** Streamlit Cloud or Render — see “Option C” above |

So: **yes, you do this in the terminal**—either the one in Cursor/VS Code (with the project folder as current directory) or a terminal on your server after you’re in the project folder there.
