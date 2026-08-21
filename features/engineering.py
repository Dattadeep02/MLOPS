import numpy as np
import pandas as pd


def encode_list_column(df, column):
    """
    Convert a column containing lists into multi-hot encoded features.

    Example:
        ["Action", "RPG"]

    becomes:
        genres_Action = 1
        genres_RPG = 1
    """

    encoded = df[column].explode().str.get_dummies()

    encoded = encoded.groupby(level=0).max()

    encoded.columns = [
        f"{column}_{str(col).strip().replace(' ', '_')}"
        for col in encoded.columns
    ]

    return encoded


def create_features(df):
    """
    Create the feature matrix X and target y.

    Features:
        - price
        - genres
        - tags

    Target:
        - log-transformed estimated owners
    """

    df = df.copy()

    # --------------------------------
    # Handle missing price
    # --------------------------------

    price_median = df["price"].median()

    df["price"] = df["price"].fillna(price_median)

    # --------------------------------
    # Encode genres
    # --------------------------------

    genre_features = encode_list_column(
        df,
        "genres"
    )

    # --------------------------------
    # Encode tags
    # --------------------------------

    tag_features = encode_list_column(
        df,
        "tags"
    )

    # --------------------------------
    # Keep price
    # --------------------------------

    price_feature = df[["price"]].copy()

    # --------------------------------
    # Combine features
    # --------------------------------

    X = pd.concat(
        [
            price_feature,
            genre_features,
            tag_features
        ],
        axis=1
    )

    # --------------------------------
    # Target
    # --------------------------------

    y = np.log1p(
        df["estimated_owners"]
    )

    return X, y