import os

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from data.data_loader import load_games
from features.parsing import clean_data
from features.engineering import create_features


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a regression model using MAE and R².
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"\n===== {model_name} =====")
    print(f"MAE: {mae:.4f}")
    print(f"R² : {r2:.4f}")

    return {
        "model": model_name,
        "mae": mae,
        "r2": r2
    }


def train_models():
    """
    Load data, prepare features, split the data,
    train baseline and Random Forest models.
    """

    # --------------------------------
    # Load data
    # --------------------------------

    games = load_games()

    # --------------------------------
    # Clean data
    # --------------------------------

    games = clean_data(games)

    # --------------------------------
    # Create ML features
    # --------------------------------

    X, y = create_features(games)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target shape:   {y.shape}")

    # --------------------------------
    # Train-test split
    # --------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # --------------------------------
    # Baseline model
    # --------------------------------

    baseline = DummyRegressor(
        strategy="mean"
    )

    baseline.fit(
        X_train,
        y_train
    )

    baseline_results = evaluate_model(
        baseline,
        X_test,
        y_test,
        "Dummy Baseline"
    )

    # --------------------------------
    # Random Forest
    # --------------------------------

    random_forest = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    random_forest.fit(
        X_train,
        y_train
    )

    rf_results = evaluate_model(
        random_forest,
        X_test,
        y_test,
        "Random Forest"
    )

    # --------------------------------
    # Save model
    # --------------------------------

    os.makedirs(
        "model/saved",
        exist_ok=True
    )

    joblib.dump(
        random_forest,
        "model/saved/random_forest.pkl"
    )

    print(
        "\nModel saved to "
        "model/saved/random_forest.pkl"
    )

    return (
        random_forest,
        X_test,
        y_test,
        baseline_results,
        rf_results
    )


if __name__ == "__main__":
    train_models()