import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from data.data_loader import load_games
from features.parsing import clean_data


# ============================================================
# Configuration
# ============================================================

PREDICTIONS_PATH = "model/saved/test_predictions.csv"
OUTPUT_PATH = "model/saved/final_demand_opportunities.csv"

MIN_GAMES = 30

CI_LEVEL = 0.95
FDR_ALPHA = 0.05

STRONG_POSITIVE_RATE = 0.60
PROMISING_POSITIVE_RATE = 0.50


# ============================================================
# Load prediction data
# ============================================================

def load_prediction_data():
    """
    Load the model's test predictions.

    The original dataframe index was saved by pandas as
    'Unnamed: 0'. Recover it as game_index so that predictions
    can be matched with the cleaned games dataframe.
    """

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    predictions = predictions.rename(
        columns={
            "Unnamed: 0": "game_index"
        }
    )

    return predictions


# ============================================================
# Load and clean games
# ============================================================

def prepare_game_data():
    """
    Load and clean the original Steam games data.
    """

    games = load_games()

    games = clean_data(games)

    return games


# ============================================================
# Create tag pairs
# ============================================================

def create_tag_pairs(tags):
    """
    Create all unique pairs of tags for one game.

    Example:

        ["Action", "RPG", "Indie"]

    becomes:

        ("Action", "Indie")
        ("Action", "RPG")
        ("Indie", "RPG")
    """

    if not isinstance(tags, list):
        return []

    tags = sorted(
        set(
            tag.strip()
            for tag in tags
            if isinstance(tag, str) and tag.strip()
        )
    )

    if len(tags) < 2:
        return []

    return list(
        itertools.combinations(tags, 2)
    )


# ============================================================
# Create game-level tag pairs
# ============================================================

def create_game_tag_pairs(games, predictions):
    """
    Match predictions to cleaned games and create one row
    for every game-tag-pair combination.
    """

    results = predictions.merge(
        games[["tags"]],
        left_on="game_index",
        right_index=True,
        how="inner"
    )

    print(
        f"Games matched with predictions: "
        f"{len(results)}"
    )

    results["tag_pairs"] = results["tags"].apply(
        create_tag_pairs
    )

    results = results.explode(
        "tag_pairs"
    )

    results = results[
        results["tag_pairs"].notna()
        & (results["tag_pairs"] != "")
    ]

    results["tag_1"] = results[
        "tag_pairs"
    ].apply(lambda x: x[0])

    results["tag_2"] = results[
        "tag_pairs"
    ].apply(lambda x: x[1])

    print(
        f"Game-tag-pair rows: "
        f"{len(results)}"
    )

    return results


# ============================================================
# Calculate statistics
# ============================================================

def calculate_tag_statistics(tag_pairs):
    """
    Calculate statistical properties of the demand gap
    for every tag combination.
    """

    print("\nCalculating tag statistics...")

    grouped = tag_pairs.groupby(
        ["tag_1", "tag_2"]
    )

    statistics = []

    for (tag_1, tag_2), group in grouped:

        gaps = group[
            "demand_gap"
        ].dropna()

        n = len(gaps)

        if n < MIN_GAMES:
            continue

        mean_gap = gaps.mean()

        median_gap = gaps.median()

        std_gap = gaps.std(
            ddof=1
        )

        positive_gap_rate = (
            gaps > 0
        ).mean()

        # ----------------------------------------------------
        # One-sample t-test
        #
        # H0: average demand gap = 0
        # H1: average demand gap != 0
        # ----------------------------------------------------

        if std_gap == 0 or np.isnan(std_gap):

            p_value = 1.0

        else:

            _, p_value = stats.ttest_1samp(
                gaps,
                popmean=0
            )

        # ----------------------------------------------------
        # Confidence interval
        # ----------------------------------------------------

        standard_error = (
            std_gap / np.sqrt(n)
        )

        t_critical = stats.t.ppf(
            1 - (1 - CI_LEVEL) / 2,
            df=n - 1
        )

        margin_error = (
            t_critical * standard_error
        )

        ci_lower = (
            mean_gap - margin_error
        )

        ci_upper = (
            mean_gap + margin_error
        )

        statistics.append(
            {
                "tag_1": tag_1,
                "tag_2": tag_2,
                "games": n,
                "avg_gap": mean_gap,
                "median_gap": median_gap,
                "std_gap": std_gap,
                "positive_gap_rate": positive_gap_rate,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    results = pd.DataFrame(
        statistics
    )

    print(
        f"Tag combinations tested: "
        f"{len(results)}"
    )

    return results


# ============================================================
# FDR correction
# ============================================================

def apply_fdr_correction(results):
    """
    Apply Benjamini-Hochberg False Discovery Rate correction.
    """

    print(
        "\nApplying Benjamini-Hochberg "
        "FDR correction..."
    )

    rejected, adjusted_p_values, _, _ = (
        multipletests(
            results["p_value"],
            alpha=FDR_ALPHA,
            method="fdr_bh"
        )
    )

    results["adjusted_p_value"] = (
        adjusted_p_values
    )

    results["fdr_significant"] = (
        rejected
    )

    return results


# ============================================================
# Opportunity score
# ============================================================

def calculate_opportunity_score(results):
    """
    Rank demand opportunities.

    The score considers:

    1. Average positive demand gap
    2. Percentage of games beating prediction
    3. Number of games supporting the signal

    This score is for ranking, not statistical significance.
    """

    results = results.copy()

    effect_component = (
        results["avg_gap"]
        .clip(lower=0)
    )

    consistency_component = (
        results["positive_gap_rate"]
    )

    sample_component = np.sqrt(
        results["games"]
        / (
            results["games"] + 100
        )
    )

    results["opportunity_score"] = (
        effect_component
        * consistency_component
        * sample_component
    )

    return results


# ============================================================
# Opportunity classification
# ============================================================

def classify_opportunities(results):
    """
    Classify tag combinations based on the confidence
    interval and consistency of positive demand gaps.

    FDR significance remains separately reported.
    """

    results = results.copy()

    conditions = [

        (
            (results["ci_lower"] > 0)
            &
            (
                results["positive_gap_rate"]
                >= STRONG_POSITIVE_RATE
            )
        ),

        (
            (results["ci_lower"] > 0)
            &
            (
                results["positive_gap_rate"]
                >= PROMISING_POSITIVE_RATE
            )
        ),

        (
            results["ci_lower"] > 0
        ),
    ]

    choices = [
        "Strong Opportunity",
        "Promising Opportunity",
        "Statistical Signal",
    ]

    results["opportunity_level"] = np.select(
        conditions,
        choices,
        default="Not Supported"
    )

    return results


# ============================================================
# Diagnostics
# ============================================================

def print_diagnostics(results):

    print("\n")
    print("=" * 70)
    print("DEMAND OPPORTUNITY DIAGNOSTICS")
    print("=" * 70)

    total = len(results)

    fdr_count = int(
        results["fdr_significant"].sum()
    )

    ci_positive_count = int(
        (
            results["ci_lower"] > 0
        ).sum()
    )

    positive_average_count = int(
        (
            results["avg_gap"] > 0
        ).sum()
    )

    strong_count = int(
        (
            results["opportunity_level"]
            == "Strong Opportunity"
        ).sum()
    )

    promising_count = int(
        (
            results["opportunity_level"]
            == "Promising Opportunity"
        ).sum()
    )

    signal_count = int(
        (
            results["opportunity_level"]
            == "Statistical Signal"
        ).sum()
    )

    print(
        f"\nTotal combinations tested: "
        f"{total}"
    )

    print(
        f"FDR significant:            "
        f"{fdr_count}"
    )

    print(
        f"CI lower > 0:               "
        f"{ci_positive_count}"
    )

    print(
        f"Positive average gap:       "
        f"{positive_average_count}"
    )

    print(
        f"\nStrong opportunities:       "
        f"{strong_count}"
    )

    print(
        f"Promising opportunities:    "
        f"{promising_count}"
    )

    print(
        f"Statistical signals:        "
        f"{signal_count}"
    )


# ============================================================
# Display top opportunities
# ============================================================

def print_top_opportunities(
    results,
    n=20
):

    opportunities = results[
        results["opportunity_level"].isin(
            [
                "Strong Opportunity",
                "Promising Opportunity",
                "Statistical Signal",
            ]
        )
    ].copy()

    opportunities = opportunities.sort_values(
        "opportunity_score",
        ascending=False
    )

    print("\n")
    print("=" * 70)
    print("TOP DEMAND OPPORTUNITIES")
    print("=" * 70)

    if opportunities.empty:

        print(
            "\nNo demand opportunities found."
        )

        return

    columns = [
        "tag_1",
        "tag_2",
        "games",
        "avg_gap",
        "median_gap",
        "positive_gap_rate",
        "ci_lower",
        "ci_upper",
        "p_value",
        "adjusted_p_value",
        "opportunity_score",
        "opportunity_level",
    ]

    display_data = opportunities[
        columns
    ].head(n).copy()

    print(
        display_data.to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading games and predictions..."
    )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    predictions = load_prediction_data()

    print(
        f"Predictions: "
        f"{len(predictions)}"
    )

    # --------------------------------------------------------
    # Load cleaned games
    # --------------------------------------------------------

    games = prepare_game_data()

    print(
        f"Games: "
        f"{len(games)}"
    )

    # --------------------------------------------------------
    # Create game-level tag pairs
    # --------------------------------------------------------

    tag_pairs = create_game_tag_pairs(
        games,
        predictions
    )

    if tag_pairs.empty:

        print(
            "\nNo tag pairs were created."
        )

        return

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    results = calculate_tag_statistics(
        tag_pairs
    )

    if results.empty:

        print(
            "\nNo tag combinations passed "
            "the minimum game threshold."
        )

        return

    # --------------------------------------------------------
    # FDR correction
    # --------------------------------------------------------

    results = apply_fdr_correction(
        results
    )

    # --------------------------------------------------------
    # Opportunity score
    # --------------------------------------------------------

    results = calculate_opportunity_score(
        results
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    results = classify_opportunities(
        results
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print_diagnostics(
        results
    )

    # --------------------------------------------------------
    # Top opportunities
    # --------------------------------------------------------

    print_top_opportunities(
        results,
        n=20
    )

    # --------------------------------------------------------
    # Sort final results
    # --------------------------------------------------------

    results = results.sort_values(
        "opportunity_score",
        ascending=False
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_PATH
        ),
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)

    print(
        f"\nResults saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()