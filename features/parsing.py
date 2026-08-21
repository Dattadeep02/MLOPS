import ast
import numpy as np
import pandas as pd


def parse_owner_range(value):
    """
    Convert estimated owner ranges into a numeric midpoint.

    Example:
        '20000 - 50000' -> 35000
        '0 - 20000'     -> 10000
    """

    if pd.isna(value):
        return np.nan

    try:
        lower, upper = value.split("-")

        lower = float(lower.strip())
        upper = float(upper.strip())

        return (lower + upper) / 2

    except (ValueError, AttributeError):
        return np.nan


def parse_list(value):
    """
    Convert a string representation of a Python list
    into an actual Python list.

    Example:
        '["Action", "Indie"]'
        ->
        ["Action", "Indie"]
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

        return []

    except (ValueError, SyntaxError):
        return []


def clean_data(df):
    """
    Clean the columns required for our ML problem.

    Required columns:
        price
        genres
        tags
        estimated_owners
    """

    df = df.copy()

    # -----------------------------
    # Parse estimated owners
    # -----------------------------

    df["estimated_owners"] = (
        df["estimated_owners"]
        .apply(parse_owner_range)
    )

    # -----------------------------
    # Parse genres
    # -----------------------------

    df["genres"] = (
        df["genres"]
        .apply(parse_list)
    )

    # -----------------------------
    # Parse tags
    # -----------------------------

    df["tags"] = (
        df["tags"]
        .apply(parse_list)
    )

    # -----------------------------
    # Clean price
    # -----------------------------

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    # -----------------------------
    # Remove rows without target
    # -----------------------------

    df = df.dropna(
        subset=["estimated_owners"]
    )

    return df
