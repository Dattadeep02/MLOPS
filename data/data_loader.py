"""
data_loader.py

Purpose: Load the raw Steam dataset CSV files into pandas DataFrames.
This is intentionally "dumb" — it does NOT clean or parse anything.
Cleaning/parsing lives in features/parsing.py. Keeping load and clean
separate means you can re-run parsing without re-reading from disk,
and you can unit-test parsing logic on small fake DataFrames later.

Expected files (place these yourself in data/raw/):
    data/raw/steam_games.csv           -> main dataset (games + metadata)
    data/raw/steam_games_reviews.csv   -> critic review snippets (optional)
"""

from pathlib import Path
import pandas as pd

# Project root is two levels up from this file (MLOPS/data/data_loader.py -> MLOPS/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

GAMES_FILE = RAW_DIR / "steam_games.csv"
REVIEWS_FILE = RAW_DIR / "steam_games_reviews.csv"


def load_games(path: Path = GAMES_FILE) -> pd.DataFrame:
    """
    Load the main Steam games dataset.

    NOTE: This file is large (likely 100k+ rows) and may have encoding
    quirks (special characters in game names/descriptions). We use
    encoding_errors='replace' so a bad byte doesn't crash the whole load.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure steam_games.csv is in data/raw/"
        )

    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    print(f"[load_games] Loaded {len(df):,} rows, {len(df.columns)} columns from {path.name}")
    return df


def load_reviews(path: Path = REVIEWS_FILE) -> pd.DataFrame:
    """
    Load the critic-review-snippet dataset (app_id, name, reviews text).
    This is a smaller, optional supplementary file — only ~13k games
    have critic review coverage, so this will NOT cover every game
    in load_games(). Treat any features derived from this as
    'has_critic_coverage' style flags, not universal features.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure steam_games_reviews.csv is in data/raw/"
        )

    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    print(f"[load_reviews] Loaded {len(df):,} rows, {len(df.columns)} columns from {path.name}")
    return df


def quick_inspect(df: pd.DataFrame, name: str = "dataframe") -> None:
    """
    Print a fast sanity-check summary of a loaded DataFrame.
    Run this FIRST on your real data before writing any parsing logic —
    it tells you the actual raw format of tricky columns like
    estimated_owners, tags, genres so you're not guessing.
    """
    print(f"\n===== Inspecting: {name} =====")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values (top 15):\n", df.isnull().sum().sort_values(ascending=False).head(15))

    # Print raw sample values for the columns that usually need parsing.
    # This is the important part — check these against what I assumed
    # in features/parsing.py before running that module.
    tricky_cols = ["estimated_owners", "tags", "genres", "categories", "developers", "publishers"]
    for col in tricky_cols:
        if col in df.columns:
            print(f"\nSample raw values for '{col}':")
            print(df[col].dropna().head(3).tolist())


if __name__ == "__main__":
    # Running this file directly (python data/data_loader.py) does a
    # quick load + inspect, no modeling. Good first thing to run.
    games_df = load_games()
    quick_inspect(games_df, "steam_games.csv")

    try:
        reviews_df = load_reviews()
        quick_inspect(reviews_df, "steam_games_reviews.csv")
    except FileNotFoundError as e:
        print(e)
