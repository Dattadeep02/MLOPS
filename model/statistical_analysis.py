import numpy as np
import pandas as pd
from scipy import stats


INPUT_PATH = "model/saved/tag_demand_gaps.csv"
OUTPUT_PATH = "model/saved/statistically_validated_gaps.csv"


def calculate_statistics(df):
    """
    Calculate statistical evidence for each tag combination.

    For each tag pair we calculate:
    - standard error
    - 95% confidence interval
    - percentage of games that beat model expectation
    """

    # --------------------------------
    # Standard error of the mean
    # --------------------------------

    df["standard_error"] = (
        df["std_gap"] / np.sqrt(df["games"])
    )

    # --------------------------------
    # 95% confidence interval
    # --------------------------------

    # Degrees of freedom = n - 1
    df["t_critical"] = df["games"].apply(
        lambda n: stats.t.ppf(
            0.975,
            df=n - 1
        )
    )

    margin_of_error = (
        df["t_critical"]
        * df["standard_error"]
    )

    df["ci_lower"] = (
        df["avg_gap"]
        - margin_of_error
    )

    df["ci_upper"] = (
        df["avg_gap"]
        + margin_of_error
    )

    return df


def calculate_positive_gap_rate():
    """
    Calculate the percentage of games that beat
    their predicted ownership.

    This requires the original prediction-level data.
    """

    predictions = pd.read_csv(
        "model/saved/test_predictions.csv"
    )

    predictions["positive_gap"] = (
        predictions["demand_gap"] > 0
    )

    return predictions


def main():

    print("Loading tag demand gaps...")

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Tag combinations loaded: {len(df)}"
    )

    # --------------------------------
    # Calculate confidence intervals
    # --------------------------------

    df = calculate_statistics(df)

    # --------------------------------
    # Select statistically positive gaps
    # --------------------------------

    statistically_positive = df[
        df["ci_lower"] > 0
    ].copy()

    # --------------------------------
    # Sort strongest evidence first
    # --------------------------------

    statistically_positive = (
        statistically_positive
        .sort_values(
            by="avg_gap",
            ascending=False
        )
    )

    # --------------------------------
    # Save results
    # --------------------------------

    statistically_positive.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------
    # Display results
    # --------------------------------

    print(
        "\n============================================================"
    )

    print(
        "STATISTICALLY POSITIVE DEMAND GAPS"
    )

    print(
        "============================================================"
    )

    columns = [
        "tag_1",
        "tag_2",
        "games",
        "avg_gap",
        "median_gap",
        "std_gap",
        "ci_lower",
        "ci_upper"
    ]

    print(
        statistically_positive[
            columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print(
        f"\nTotal statistically positive combinations: "
        f"{len(statistically_positive)}"
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()