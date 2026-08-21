# Steam Game Demand Gap Prediction

An ML-based demand analysis system that predicts how many owners a Steam game
should have based on its price, genres, and tags, and identifies tag
combinations where games consistently outperform the model's expectations.

The goal is to help indie game developers identify potential demand gaps using
data-driven evidence rather than relying purely on intuition or guesswork.

---

## Project Status

🚧 **In Development**

The core ownership prediction model and initial demand-gap analysis are
implemented.

The current system includes:

- Data loading
- Data cleaning
- Feature engineering
- Log transformation of ownership estimates
- Random Forest regression
- Model evaluation
- Demand-gap calculation
- Tag-combination analysis
- Statistical validation
- Multiple-testing correction using Benjamini-Hochberg FDR

The statistical definition of a final "demand opportunity" is currently being
refined.

---

## Problem Statement

For an indie developer, deciding which type of game to build is difficult.

Popular genres and tags do not necessarily represent unexplored opportunities.
A tag may be popular but highly competitive, while a less obvious combination
of tags may consistently perform better than expected.

This project investigates the following question:

> Given a game's price, genres, and tags, how many owners should the game
> reasonably be expected to have?

The model predicts expected ownership for a game and compares that prediction
with its actual estimated ownership.

This produces a **demand gap**:

```text
Demand Gap = Actual Log Owners - Predicted Log Owners