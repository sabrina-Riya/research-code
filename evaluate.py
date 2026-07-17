"""
evaluate.py
-----------
Turns the raw per-model ModelResult objects from train.py into the structured
comparison tables reported in thesis Chapter 7 (MAE/MSE/R2/RMSE-by-country
tables, §7.2-7.5) and reproduces the percentile-based emission-severity
classification used in the notebook's "Zone Check" section (Cell 40).

This module has no knowledge of how results were produced (neural vs.
classical, which country) — it only consumes ModelResult objects and raw
emissions data, per the README's "criteria-agnostic" module design.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from train import ModelResult

METRIC_COLUMNS = ["MAE", "MSE", "RMSE", "R2"]

# Same ordered zone labels and cumulative percentile cut points as the
# notebook's Cell 40 ("Zone Check"), computed against the latest available
# emissions year (2018) across all countries in the historical dataset.
ZONE_PERCENTILES = {
    "Clean Zone": 0.10,
    "Green Zone": 0.25,
    "Safe Zone": 0.40,
    "Stable Zone": 0.50,
    "Warning Zone": 0.60,
    "Caution Zone": 0.75,
    "Risk Zone": 0.85,
    "Danger Zone": 0.90,
    "Severe Zone": 0.95,
    "Critical Zone": 1.00,  # max(), i.e. the 100th percentile
}


# --------------------------------------------------------------------------- #
# Cross-model, cross-country comparison tables (mirrors Tables 7.1-7.4)
# --------------------------------------------------------------------------- #

def build_metric_table(country_results: Dict[str, Dict[str, ModelResult]],
                        metric: str) -> pd.DataFrame:
    """
    country_results: {country_name: {model_name: ModelResult}}
    metric: one of METRIC_COLUMNS.

    Returns a DataFrame shaped like Table 7.1/7.2/7.3/7.4 in the thesis —
    rows are countries, columns are models.
    """
    metric = metric.upper()
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"metric must be one of {METRIC_COLUMNS}, got '{metric}'.")

    attr = metric.lower()
    rows = {}
    for country, model_map in country_results.items():
        rows[country] = {
            model_name: getattr(result, attr, np.nan)
            for model_name, result in model_map.items()
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Country"
    return table


def build_all_metric_tables(country_results: Dict[str, Dict[str, ModelResult]]) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper producing MAE/MSE/RMSE/R2 tables in one call."""
    return {metric: build_metric_table(country_results, metric) for metric in METRIC_COLUMNS}


def best_model_per_country(country_results: Dict[str, Dict[str, ModelResult]]) -> pd.DataFrame:
    """
    Reproduces Table 7.5 ("Predicted CO2 Emissions"): for each country, the
    model with the lowest MAE and that model's target-year prediction.
    """
    records = []
    for country, model_map in country_results.items():
        valid = {k: v for k, v in model_map.items() if np.isfinite(v.mae)}
        if not valid:
            records.append({"Country": country, "Best Model": None,
                             "Predicted Emission (MtCO2)": np.nan, "MAE": np.nan})
            continue
        best_name = min(valid, key=lambda k: valid[k].mae)
        best = valid[best_name]
        records.append({
            "Country": country,
            "Best Model": best_name,
            "Predicted Emission (MtCO2)": best.prediction,
            "MAE": best.mae,
        })
    return pd.DataFrame.from_records(records).set_index("Country")


# --------------------------------------------------------------------------- #
# Emission severity classification (Cell 40 "Zone Check", generalized to any
# reference-year distribution rather than a hardcoded '2018' column)
# --------------------------------------------------------------------------- #

def compute_zone_thresholds(reference_emissions: Iterable[float]) -> Dict[str, float]:
    """
    Compute the percentile cut points against a reference cross-country
    emissions distribution (the notebook uses the latest available year,
    2018, across all 195 countries in the historical dataset).
    """
    series = pd.Series(list(reference_emissions)).dropna()
    if series.empty:
        raise ValueError("reference_emissions is empty after dropping nulls.")

    thresholds = {}
    for zone, pct in ZONE_PERCENTILES.items():
        thresholds[zone] = series.max() if pct == 1.00 else series.quantile(pct)
    return thresholds


def classify_severity(predicted_emission: float, thresholds: Dict[str, float]) -> str:
    """
    Assigns a predicted emission value to the first zone (in ascending
    severity order) whose threshold it does not exceed — identical decision
    order to the notebook's `determine_risk_level`.
    """
    for zone in ZONE_PERCENTILES:  # dict preserves insertion order (ascending severity)
        if zone == "Critical Zone":
            return zone
        if predicted_emission <= thresholds[zone]:
            return zone
    return "Critical Zone"


def severity_table(best_model_df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """
    Attach a 'Severity Zone' column to the best_model_per_country() output —
    this is the per-country classification table referenced in README
    ("emission severity classification") and thesis §6.4/Ch.7.
    """
    out = best_model_df.copy()
    out["Severity Zone"] = out["Predicted Emission (MtCO2)"].apply(
        lambda v: classify_severity(v, thresholds) if np.isfinite(v) else None
    )
    return out
