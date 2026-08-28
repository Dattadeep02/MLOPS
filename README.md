STEAM GAME DEMAND GAP PREDICTION
================================

PROJECT OVERVIEW
----------------

This project builds a Machine Learning model that predicts how many
owners a Steam game should have based on its price, genres, and tags.

The model then compares predicted ownership with actual ownership to
identify games that perform better than expected.

The final goal is to identify tag combinations that show statistically
supported demand opportunities for indie game developers.


USE CASE
--------

A tool that predicts how many owners a game "should" get based on its
price, genre, and tags, then flags tag combinations where real games
consistently beat that prediction.

The purpose is to help indie developers identify potential demand gaps
using data instead of guesswork.


DATASET
-------

Dataset:
Steam Games Dataset - Daily Updates

Kaggle:
https://www.kaggle.com/datasets/hubertsidorowicz/steam-games-dataset-daily-updates

Required files:

data/
    steam_games.csv
    steam_games_reviews.csv

The current ML pipeline primarily uses steam_games.csv.


PROJECT STRUCTURE
-----------------

MLOPS/
|
|-- data/
|   |-- steam_games.csv
|   |-- steam_games_reviews.csv
|   `-- data_loader.py
|
|-- features/
|   |-- parsing.py
|   `-- engineering.py
|
|-- model/
|   |-- train.py
|   |-- demand_gaps.py
|   |-- statistical_analysis.py
|   `-- saved/
|
|-- .gitignore
|-- requirements.txt
`-- README.txt


PIPELINE
--------

1. Load Steam game data.
2. Clean and parse the raw data.
3. Convert genres and tags into numerical features.
4. Log-transform the ownership target because ownership is highly
   right-skewed.
5. Train a Random Forest regression model.
6. Predict expected game ownership.
7. Calculate the demand gap between actual and predicted ownership.
8. Analyze tag combinations.
9. Apply statistical validation and rank demand opportunities.


HOW TO RUN
----------

Run all commands from the MLOPS project root.

Train the model:

    python3 -m model.train

This creates:

    model/saved/random_forest.pkl
    model/saved/test_predictions.csv


Calculate demand gaps:

    python3 -m model.demand_gaps

This creates:

    model/saved/tag_demand_gaps.csv


Run statistical analysis:

    python3 -m model.statistical_analysis

This creates:

    model/saved/final_demand_opportunities.csv


MODEL PERFORMANCE
-----------------

Current Random Forest results:

    MAE: 1.2046
    R2 : 0.6924

The target variable is log-transformed because Steam ownership values
have a highly skewed distribution.


DEMAND GAP
----------

Demand gap represents the difference between actual and predicted
ownership.

    Demand Gap = Actual Ownership - Predicted Ownership

A positive demand gap means that a game performed better than the model
expected given its available features.

Tag combinations with consistently positive gaps are investigated as
potential demand opportunities.


STATISTICAL ANALYSIS
--------------------

The project evaluates tag combinations using:

- Average demand gap
- Median demand gap
- Positive gap rate
- Confidence intervals
- Statistical significance
- Benjamini-Hochberg FDR correction
- Opportunity scoring

The statistical analysis is intended to identify demand signals.

A statistically positive relationship does NOT prove that a particular
tag combination causes higher game ownership.


CURRENT STATUS
--------------

Data loading                 [DONE]
Data cleaning                [DONE]
Feature engineering         [DONE]
Random Forest model          [DONE]
Demand-gap analysis          [DONE]
Statistical validation      [DONE]
Demand opportunity ranking  [DONE]

Future work:

- Improve model performance
- Experiment with additional ML algorithms
- Improve feature engineering
- Validate results on newer data
- Build a user-facing prediction tool
- Deploy the model/API


IMPORTANT
---------

The dataset and generated model files may be large and should not
normally be committed to GitHub.

The .gitignore file is used to prevent unnecessary datasets, Python
cache files, and generated model artifacts from being pushed to the
repository.


GIT WORKFLOW
------------

The project uses feature branches for isolated development and testing.

Create a new feature branch:

    git checkout -b feature/<feature-name>

Make the required changes and check the status:

    git status

Stage the changes:

    git add .

Commit the changes:

    git commit -m "feat: describe the change"

Push the feature branch:

    git push -u origin feature/<feature-name>

Changes can then be reviewed and merged into the main branch.


REBASE DEMONSTRATION
--------------------

This section demonstrates Git rebase workflow.


RESET DEMONSTRATION
-------------------

This temporary change is used to demonstrate git reset.


INTERACTIVE REBASE DEMONSTRATION
--------------------------------

This section demonstrates interactive rebase in Git.
Interactive rebase can be used to clean up commit history and modify commit messages.


CURRENT DEVELOPMENT
-------------------

The project is currently being developed through feature branches.
Recent work includes improvements to model training configuration and
statistical demand-opportunity analysis.