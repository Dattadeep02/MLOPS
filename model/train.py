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
        "r2": r2,
        "predictions": predictions
    }


def train_models():
    """
    Load data, prepare features, split the data,
    train baseline and Random Forest models,
    and save test predictions.
    """

    # --------------------------------
    # 1. Load data
    # --------------------------------

    games = load_games()

    # --------------------------------
    # 2. Clean data
    # --------------------------------

    games = clean_data(games)

    # --------------------------------
    # 3. Create ML features
    # --------------------------------

    X, y = create_features(games)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target shape:   {y.shape}")

    # --------------------------------
    # 4. Train-test split
    # --------------------------------

    train_indices, test_indices = train_test_split(
        X.index,
        test_size=0.20,
        random_state=42
    )

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]

    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # --------------------------------
    # 5. Dummy baseline
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
    # 6. Random Forest
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
    # 7. Save trained model
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

    # --------------------------------
    # 8. Create demand-gap data
    # --------------------------------

    prediction_data = pd.DataFrame({
        "actual_log_owners": y_test,
        "predicted_log_owners": rf_results["predictions"]
    })

    prediction_data["demand_gap"] = (
        prediction_data["actual_log_owners"]
        - prediction_data["predicted_log_owners"]
    )

    # --------------------------------
    # 9. Save test predictions
    # --------------------------------

    prediction_data.to_csv(
        "model/saved/test_predictions.csv"
    )

    print(
        "Test predictions saved to "
        "model/saved/test_predictions.csv"
    )

    return (
        random_forest,
        X_test,
        y_test,
        baseline_results,
        rf_results
    )


if __name__ == "__main__":
    train_models()new cchanges 
