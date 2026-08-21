import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from data.data_loader import load_games
from features.parsing import clean_data


PREDICTIONS_PATH = "model/saved/test_predictions.csv"
OUTPUT_PATH = "model/saved/final_demand_opportunities.csv"

MIN_GAMES = 30
MIN_POSITIVE_RATE = 0.60
ALPHA = 0.05


def load_data():
    """
    Load cleaned game data and model predictions.
    """

    games = load_games()
    games = clean_data(games)

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    predictions = predictions.rename(
        columns={
            "Unnamed: 0": "game_index"
        }
    )

    return games, predictions


def create_tag_pairs(tags):
    """
    Create unique combinations of two tags.

    Example:

    ["Action", "RPG", "Indie"]

    becomes:

    Action + RPG
    Action + Indie
    Indie + RPG
    """

    if not isinstance(tags, list):
        return []

    tags = sorted(
        set(
            tag.strip()
            for tag in tags
            if isinstance(tag, str)
            and tag.strip()
        )
    )

    return list(
        itertools.combinations(tags, 2)
    )


def build_tag_dataset(games, predictions):
    """
    Match model predictions with games and
    create game-level tag-pair observations.
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

    results = results.dropna(
        subset=["tag_pairs"]
    )

    results["tag_1"] = results[
        "tag_pairs"
    ].apply(lambda x: x[0])

    results["tag_2"] = results[
        "tag_pairs"
    ].apply(lambda x: x[1])

    return results


def calculate_tag_statistics(tag_data):
    """
    Calculate statistics for every tag pair.
    """

    grouped = []

    for (tag_1, tag_2), group in tag_data.groupby(
        ["tag_1", "tag_2"]
    ):

        gaps = group["demand_gap"]

        n = len(gaps)

        if n < MIN_GAMES:
            continue

        mean_gap = gaps.mean()
        median_gap = gaps.median()
        std_gap = gaps.std()

        positive_rate = (
            (gaps > 0).mean()
        )

        # --------------------------------
        # One-sample t-test
        #
        # H0: mean demand gap = 0
        # H1: mean demand gap > 0
        # --------------------------------

        t_statistic, p_value_two_sided = (
            stats.ttest_1samp(
                gaps,
                0
            )
        )

        if t_statistic > 0:
            p_value = (
                p_value_two_sided / 2
            )
        else:
            p_value = 1.0

        # --------------------------------
        # 95% confidence interval
        # --------------------------------

        standard_error = (
            std_gap / np.sqrt(n)
        )

        t_critical = stats.t.ppf(
            0.975,
            df=n - 1
        )

        margin_of_error = (
            t_critical
            * standard_error
        )

        ci_lower = (
            mean_gap
            - margin_of_error
        )

        ci_upper = (
            mean_gap
            + margin_of_error
        )

        grouped.append({
            "tag_1": tag_1,
            "tag_2": tag_2,
            "games": n,
            "avg_gap": mean_gap,
            "median_gap": median_gap,
            "std_gap": std_gap,
            "positive_gap_rate": positive_rate,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

    return pd.DataFrame(grouped)


def apply_fdr_correction(df):
    """
    Apply Benjamini-Hochberg False Discovery Rate
    correction to all tag-pair statistical tests.
    """

    if df.empty:
        return df

    rejected, adjusted_pvalues, _, _ = (
        multipletests(
            df["p_value"],
            alpha=ALPHA,
            method="fdr_bh"
        )
    )

    df["adjusted_p_value"] = (
        adjusted_pvalues
    )

    df["fdr_significant"] = (
        rejected
    )

    return df


def print_diagnostics(df):
    """
    Show where tag combinations are being
    eliminated by our statistical filters.
    """

    print(
        "\n================ DIAGNOSTICS ================"
    )

    print(
        f"\nTotal combinations tested: "
        f"{len(df)}"
    )

    print(
        "\nFDR significant:",
        df["fdr_significant"].sum()
    )

    print(
        "CI lower > 0:",
        (
            df["ci_lower"] > 0
        ).sum()
    )

    print(
        "Positive average gap:",
        (
            df["avg_gap"] > 0
        ).sum()
    )

    print(
        "Positive gap rate >= 60%:",
        (
            df["positive_gap_rate"] >= 0.60
        ).sum()
    )

    print(
        "Positive gap rate >= 55%:",
        (
            df["positive_gap_rate"] >= 0.55
        ).sum()
    )

    print(
        "Positive gap rate >= 50%:",
        (
            df["positive_gap_rate"] >= 0.50
        ).sum()
    )

    # --------------------------------
    # Sequential filtering
    # --------------------------------

    fdr_df = df[
        df["fdr_significant"]
    ]

    ci_df = fdr_df[
        fdr_df["ci_lower"] > 0
    ]

    consistency_df = ci_df[
        ci_df["positive_gap_rate"]
        >= MIN_POSITIVE_RATE
    ]

    print(
        "\nSequential filtering:"
    )

    print(
        f"After FDR correction: "
        f"{len(fdr_df)}"
    )

    print(
        f"After CI > 0: "
        f"{len(ci_df)}"
    )

    print(
        f"After {MIN_POSITIVE_RATE:.0%} "
        f"consistency requirement: "
        f"{len(consistency_df)}"
    )

    # --------------------------------
    # Inspect best FDR results
    # --------------------------------

    print(
        "\nTop FDR-significant combinations:"
    )

    diagnostic_columns = [
        "tag_1",
        "tag_2",
        "games",
        "avg_gap",
        "positive_gap_rate",
        "p_value",
        "adjusted_p_value",
        "ci_lower",
        "ci_upper"
    ]

    print(
        df
        .sort_values(
            "adjusted_p_value"
        )
        [diagnostic_columns]
        .head(20)
        .to_string(index=False)
    )


def identify_opportunities(df):
    """
    Select tag combinations that satisfy
    all statistical requirements.
    """

    opportunities = df[
        (df["fdr_significant"])
        &
        (df["ci_lower"] > 0)
        &
        (
            df["positive_gap_rate"]
            >= MIN_POSITIVE_RATE
        )
    ].copy()

    if opportunities.empty:
        return opportunities

    # --------------------------------
    # Opportunity score
    # --------------------------------
    #
    # This is a ranking score.
    # It is NOT a probability.
    #
    # Higher means:
    # - larger demand gap
    # - more consistent positive gaps
    # - stronger statistical evidence
    #

    opportunities["opportunity_score"] = (
        opportunities["avg_gap"]
        *
        opportunities["positive_gap_rate"]
        *
        (
            -np.log10(
                opportunities["adjusted_p_value"]
            )
        )
    )

    opportunities = opportunities.sort_values(
        by="opportunity_score",
        ascending=False
    )

    return opportunities


def main():

    print(
        "Loading games and predictions..."
    )

    games, predictions = load_data()

    print(
        f"Games: {len(games)}"
    )

    print(
        f"Predictions: {len(predictions)}"
    )

    # --------------------------------
    # Create game-level tag pairs
    # --------------------------------

    print(
        "\nCreating game-level tag pairs..."
    )

    tag_data = build_tag_dataset(
        games,
        predictions
    )

    print(
        f"Game-tag-pair rows: "
        f"{len(tag_data)}"
    )

    # --------------------------------
    # Calculate statistics
    # --------------------------------

    print(
        "\nCalculating tag statistics..."
    )

    statistics_df = calculate_tag_statistics(
        tag_data
    )

    print(
        f"Tag combinations tested: "
        f"{len(statistics_df)}"
    )

    # --------------------------------
    # FDR correction
    # --------------------------------

    print(
        "\nApplying Benjamini-Hochberg "
        "FDR correction..."
    )

    statistics_df = apply_fdr_correction(
        statistics_df
    )

    # --------------------------------
    # Diagnostics
    # --------------------------------

    print_diagnostics(
        statistics_df
    )

    # --------------------------------
    # Identify final opportunities
    # --------------------------------

    opportunities = identify_opportunities(
        statistics_df
    )

    # --------------------------------
    # Display final results
    # --------------------------------

    print(
        "\n=============================================================="
    )

    print(
        "STATISTICALLY VALIDATED DEMAND OPPORTUNITIES"
    )

    print(
        "=============================================================="
    )

    columns = [
        "tag_1",
        "tag_2",
        "games",
        "avg_gap",
        "positive_gap_rate",
        "adjusted_p_value",
        "ci_lower",
        "ci_upper",
        "opportunity_score"
    ]

    if opportunities.empty:

        print(
            "\nNo combinations passed all "
            "statistical requirements."
        )

    else:

        print(
            opportunities[
                columns
            ]
            .head(20)
            .to_string(index=False)
        )

    print(
        f"\nValidated opportunities: "
        f"{len(opportunities)}"
    )

    # --------------------------------
    # Save final results
    # --------------------------------

    opportunities.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()