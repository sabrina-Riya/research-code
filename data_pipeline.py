"""
data_pipeline.py
-----------------


This module owns exactly two responsibilities, per the "one job, no downstream
knowledge" design principle in the project README:

  1. Emissions-series preparation for the six time-series models (long-format
     reshape, MinMax scaling, sliding-window construction) — generalizing the
     preprocessing performed inline across the notebook's per-model cells
     (e.g. `prepare_data_for_arima`, the LSTM windowing block).
  2. The scale-invariant (L2 vector) normalization core used to make cost,
     distance, and capacity commensurable before any MCDA ranking — this is
     the fix for the "dimension washout" problem documented in README §2.

`data_pipeline.py` does not know whether its output feeds TOPSIS, a neural
model, or a plot. It receives raw tabular data and returns clean, scale-safe
arrays/matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------------------------------- #
# 1. Emissions series preparation (shared by ARIMA / SARIMA / Prophet /
#    LSTM / Bi-LSTM / GRU)
# --------------------------------------------------------------------------- #

EMISSION_YEAR_RANGE = range(1990, 2019)  # matches the notebook's fixed 1990-2018 window


def wide_to_long_emissions(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces the notebook's wide -> long reshape (Cell 3 of
    research_model_code.ipynb): drop the aggregate "World" row, keep only the
    Country + 1990-2018 year columns, melt to (Country, Year, Emission), coerce
    to numeric, and drop unusable rows.

    Parameters
    ----------
    raw : DataFrame in the CAIT historical-emissions wide format
          (one row per country, one column per year 1990-2018).

    Returns
    -------
    long_format : DataFrame with columns [Country, Year, Emission], numeric,
                  null-free, ready for per-country windowing.
    """
    data = raw.reset_index(drop=True)

    # The notebook drops row 0 (the "World" aggregate) before reshaping.
    if len(data) and str(data.iloc[0].get("Country", "")).strip().lower() == "world":
        data = data.drop(index=0).reset_index(drop=True)

    year_cols = [str(y) for y in EMISSION_YEAR_RANGE]
    missing = [c for c in year_cols if c not in data.columns]
    if missing:
        raise KeyError(
            f"Expected year columns {missing} not present in the emissions "
            "dataset — check that the 1990-2018 CAIT export was passed unmodified."
        )

    filtered = data[["Country"] + year_cols]
    long_format = filtered.melt(id_vars=["Country"], var_name="Year", value_name="Emission")
    long_format["Year"] = long_format["Year"].astype(int)
    long_format["Emission"] = pd.to_numeric(long_format["Emission"], errors="coerce")
    long_format = long_format.dropna().reset_index(drop=True)
    return long_format


def get_country_series(long_format: pd.DataFrame, country_name: str) -> np.ndarray:
    """Return a single country's emissions, sorted by year, as a 1-D array."""
    series = (
        long_format[long_format["Country"] == country_name]
        .sort_values("Year")["Emission"]
        .values
    )
    if series.size == 0:
        raise ValueError(f"No emissions records found for '{country_name}'.")
    return series


@dataclass
class WindowedDataset:
    """Container for a scaled, windowed train/val/test split."""
    scaler: MinMaxScaler
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def reshaped_for_rnn(self):
        """(batch, time_steps, features=1) view expected by Keras recurrent layers."""
        return (
            self.X_train.reshape(-1, self.X_train.shape[1], 1),
            self.X_val.reshape(-1, self.X_val.shape[1], 1),
            self.X_test.reshape(-1, self.X_test.shape[1], 1),
        )


def build_windowed_dataset(
    series: np.ndarray,
    sequence_length: int = 3,
    train_frac: float = 0.8,
    val_frac_of_train: float = 0.2,
) -> WindowedDataset:
    """
    Scale a single country's emission series to [0, 1] with MinMaxScaler and
    slide a fixed-length window across it to build 1-step supervised pairs:
    (year_t-n ... year_t-1) -> year_t.

    Matches the notebook's actual sequence_length=3 (used identically inside
    `predict_emission_recursive_lstm/_bilstm/_gru`), an 80/20 train/test split
    followed by an 80/20 split of the training partition into train/val.

    Raises
    ------
    ValueError if the series is too short to form even one window plus a
    train/val/test split — this guards the exact failure mode the notebook's
    recursive-forecast helpers already check for ("Insufficient data for
    {country_name}").
    """
    if len(series) <= sequence_length:
        raise ValueError(
            f"Series length {len(series)} does not exceed sequence_length="
            f"{sequence_length}; cannot construct a single training window."
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i : i + sequence_length])
        y.append(scaled[i + sequence_length])
    X, y = np.array(X), np.array(y)

    if len(X) < 3:
        raise ValueError(
            "Fewer than 3 windowed samples available after slicing — country "
            "series is too short for a train/val/test split."
        )

    split = int(len(X) * train_frac)
    X_train_full, y_train_full = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    if len(X_test) == 0:
        # guarantee a non-empty test partition on very short series
        X_train_full, X_test = X_train_full[:-1], X_train_full[-1:]
        y_train_full, y_test = y_train_full[:-1], y_train_full[-1:]

    val_split = max(1, int(len(X_train_full) * (1 - val_frac_of_train)))
    X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
    X_val, y_val = X_train_full[val_split:], y_train_full[val_split:]
    if len(X_val) == 0:
        X_train, X_val = X_train[:-1], X_train[-1:]
        y_train, y_val = y_train[:-1], y_train[-1:]

    return WindowedDataset(scaler, X_train, y_train, X_val, y_val, X_test, y_test)


def recursive_forecast(model, scaler: MinMaxScaler, series: np.ndarray,
                        start_year: int, target_year: int,
                        sequence_length: int = 3) -> float:
    """
    Generalized version of the notebook's `predict_emission_recursive_lstm` /
    `_bilstm` / `_gru` — one function shared by all three recurrent
    architectures, since the recursion logic (README §3) is architecture-
    agnostic: only `model.predict` differs.

    Rolls the 1-step model forward year-by-year, feeding each prediction back
    in as part of the next window (error-compounding behavior documented in
    README §3), until `target_year` is reached.
    """
    if target_year <= start_year:
        raise ValueError(f"target_year ({target_year}) must be after start_year ({start_year}).")

    normalized = scaler.transform(series.reshape(-1, 1)).flatten()
    if len(normalized) < sequence_length:
        raise ValueError("Insufficient history to seed the recursive window.")

    year = start_year
    while year < target_year:
        window = normalized[-sequence_length:].reshape(1, sequence_length, 1)
        next_val = model.predict(window, verbose=0).flatten()[0]
        normalized = np.append(normalized, next_val)
        year += 1

    return float(scaler.inverse_transform([[normalized[-1]]]).flatten()[0])


# --------------------------------------------------------------------------- #
# 2. Currency harmonization (CCS cost column -> USD)
# --------------------------------------------------------------------------- #

# Same historical-average conversion table used in the notebook's currency
# harmonization cell (average historical rates at time of thesis submission).
USD_CONVERSION_RATES = {
    "US Dollar": 1.0,
    "Australian Dollar": 0.75,
    "Canadian Dollar": 0.80,
    "Euros": 1.18,
    "British Pound": 1.35,
    "Norwegian Kroner": 0.12,
    "Danish Krone": 0.16,
    "Japenese Yen": 0.009,  # kept as spelled in the source dataset's Currency Name column
    "Chinese Yuan": 0.15,
    "Brazilian Real": 0.20,
}


def harmonize_currency(ccs_df: pd.DataFrame,
                        cost_col: str = "Cost",
                        currency_col: str = "Currency Name",
                        rates: dict | None = None) -> pd.DataFrame:
    """
    Convert every project's cost to USD in place of its native currency.
    Zero or missing costs are passed through unchanged — per README's "Known
    limitations", a $0 entry means undisclosed data, not free storage, and
    must not be silently converted into a fabricated non-zero USD figure.
    """
    rates = rates or USD_CONVERSION_RATES
    out = ccs_df.copy()

    def _convert(row):
        cost = row[cost_col]
        if pd.isna(cost) or cost == 0:
            return cost
        rate = rates.get(row[currency_col])
        return cost * rate if rate is not None else cost

    out[cost_col] = out.apply(_convert, axis=1)
    out[currency_col] = "US Dollar"
    return out


# --------------------------------------------------------------------------- #
# 3. Scale-invariant normalization core (README §2)
# --------------------------------------------------------------------------- #

def l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Column-wise vector (L2) normalization: r_ij = x_ij / sqrt(sum_k x_kj^2).

    After this transform every column is a unit vector (||column||_2 == 1),
    which is what resolves the "scale collapse" described in README §2 —
    cost in the billions and distance in the low thousands become
    commensurable, contributing to any downstream Euclidean/weighted score
    in proportion to their own cross-candidate variance, not their raw
    magnitude.

    A zero-norm column (degenerate, e.g. every candidate has identical cost)
    is left as zeros rather than raising a ZeroDivisionError, since a
    constant criterion carries no discriminating information for ranking
    anyway.
    """
    matrix = np.asarray(matrix, dtype=float)
    norms = np.sqrt((matrix ** 2).sum(axis=0))
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / safe_norms
    return normalized


def minmax_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Column-wise min-max normalization: (x - min) / (max - min).

    Provided separately (not merged into l2_normalize_matrix) because the
    original notebook's WSM function uses this normalization independently
    of TOPSIS's L2 approach — see README §2, point 1. Degenerate
    (max == min) columns are mapped to zero rather than dividing by zero.
    """
    matrix = np.asarray(matrix, dtype=float)
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    span = col_max - col_min
    safe_span = np.where(span == 0, 1.0, span)
    normalized = (matrix - col_min) / safe_span
    normalized = np.where(span == 0, 0.0, normalized)
    return normalized


def build_decision_matrix(sites_df: pd.DataFrame, criteria: Iterable[str]) -> np.ndarray:
    """
    Extract a criteria matrix (rows=candidate sites, cols=criteria) as float64,
    guarding against non-numeric or missing entries reaching the optimizer.
    """
    matrix = sites_df[list(criteria)].astype(float).to_numpy()
    if np.isnan(matrix).any():
        raise ValueError(
            "Decision matrix contains NaN values after casting to float — "
            "filter or impute missing criteria values before ranking."
        )
    return matrix
