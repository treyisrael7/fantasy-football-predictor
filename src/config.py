"""
App config from environment. Used by web app and scripts.
The app runs from CSV only; no database or cache required.
"""
from dotenv import load_dotenv

load_dotenv()
