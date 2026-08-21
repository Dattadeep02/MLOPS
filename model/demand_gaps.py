import itertools

import pandas as pd

from data.data_loader import load_games
from features.parsing import clean_data


PREDICTIONS_PATH = "model/saved/test_predictions.csv"


def load_prediction_data():
    """
    Load the model's test predictions.
    """

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    # The original DataFrame index was saved
    # by to_csv(), so recover it as game_index.
    predictions = predictions.rename(
        columns={
            "Unnamed: 0": "game_index"
        }
    )

    return predictions


def prepare_game_data():
    """
    Load and clean the original Steam games data.
    """

    games = load_games()

    games = clean_data(games)

    return games


def create_tag_pairs(tags):
    """
    Create all unique pairs of tags for one game.

    Example:

        ["Action", "RPG", "Indie"]

    becomes:

        ("Action", "RPG")
        ("Action", "Indie")
        ("Indie", "RPG")
    """

    if not isinstance(tags, list):
        return []

    # Remove duplicates and empty values
    tags = sorted(
        set(
            tag.strip()
            for tag in tags
            if isinstance(tag, str) and tag.strip()
        )
    )

    return list(
        itertools.combinations(tags, 2)
    )


def calculate_tag_gaps(games, predictions, min_games=30):
    """
    Calculate average demand gaps for tag pairs.

    Only tag pairs appearing in at least
    min_games games are considered.
    """

    # --------------------------------
    # Connect predictions to games
    # --------------------------------

    results = predictions.merge(
        games[["tags"]],
        left_on="game_index",
        right_index=True,
        how="inner"
    )

    print(
        f"\nGames matched with predictions: "
        f"{len(results)}"
    )

    # --------------------------------
    # Create tag pairs
    # --------------------------------

    results["tag_pairs"] = results["tags"].apply(
        create_tag_pairs
    )

    # --------------------------------
    # One row per tag pair
    # --------------------------------

    results = results.explode(
        "tag_pairs"
    )

    results = results.dropna(
        subset=["tag_pairs"]
    )

    # --------------------------------
    # Convert tuple into two tags
    # --------------------------------

    results["tag_1"] = results[
        "tag_pairs"
    ].apply(lambda x: x[0])

    results["tag_2"] = results[
        "tag_pairs"
    ].apply(lambda x: x[1])

    # --------------------------------
    # Aggregate
    # --------------------------------

    tag_stats = (
        results
        .groupby(["tag_1", "tag_2"])
        .agg(
            games=("demand_gap", "count"),
            avg_gap=("demand_gap", "mean"),
            median_gap=("demand_gap", "median"),
            std_gap=("demand_gap", "std")
        )
        .reset_index()
    )

    # --------------------------------
    # Minimum sample requirement
    # --------------------------------

    tag_stats = tag_stats[
        tag_stats["games"] >= min_games
    ]

    # --------------------------------
    # Rank by average demand gap
    # --------------------------------

    tag_stats = tag_stats.sort_values(
        by="avg_gap",
        ascending=False
    )

    return tag_stats


def main():

    print("Loading predictions...")

    predictions = load_prediction_data()

    print(
        f"Predictions loaded: "
        f"{len(predictions)}"
    )

    print("\nLoading game data...")

    games = prepare_game_data()

    print(
        f"Games loaded: "
        f"{len(games)}"
    )

    print("\nCalculating tag demand gaps...")

    tag_stats = calculate_tag_gaps(
        games,
        predictions,
        min_games=30
    )

    # --------------------------------
    # Display results
    # --------------------------------

    print(
        "\n================================================"
    )
    print(
        "TOP DEMAND GAP TAG COMBINATIONS"
    )
    print(
        "================================================"
    )

    print(
        tag_stats.head(20).to_string(
            index=False
        )
    )

    # --------------------------------
    # Save results
    # --------------------------------

    output_path = (
        "model/saved/tag_demand_gaps.csv"
    )

    tag_stats.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nResults saved to {output_path}"
    )


if __name__ == "__main__":
    main()