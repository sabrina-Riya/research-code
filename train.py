"""
train.py
--------
Orchestrates training of all six emission-forecasting models (LSTM, Bi-LSTM,
GRU, Prophet, ARIMA, SARIMA) for a given country, selects the best model by
MAE (matching the thesis's selection rule, §6.4/§7.2), and returns the
target-year prediction.

TensorFlow is imported lazily inside the neural-model functions only, so the
three classical models (ARIMA/SARIMA/Prophet) run unmodified in environments
where TensorFlow is not installed — this mirrors the same lazy-import
guarantee the README makes for the package as a whole.

Model hyperparameters below are pinned to the exact values used in
research_model_code.ipynb (units=50, dense=25, dropout=0.2, epochs=20,
batch_size=32, sequence_length=3, ARIMA order=(5,1,0)) so that results here
are directly comparable to the thesis's reported MAE/MSE/RMSE/R2 tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_pipeline import (
    build_windowed_dataset,
    get_country_series,
    recursive_forecast,
)

SEQUENCE_LENGTH = 3
ARIMA_ORDER = (5, 1, 0)
RNN_UNITS = 50
DENSE_UNITS = 25
DROPOUT_RATE = 0.2
EPOCHS = 20
BATCH_SIZE = 32


@dataclass
class ModelResult:
    name: str
    mae: float
    prediction: float
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Classical statistical baselines (no TensorFlow dependency)
# --------------------------------------------------------------------------- #

def train_arima(country_name: str, long_format: pd.DataFrame, target_year: int,
                 save_path: str = "arima_models") -> ModelResult:
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.preprocessing import MinMaxScaler

    series = get_country_series(long_format, country_name)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    split = int(len(normalized) * 0.8)
    train_data, val_data = normalized[:split], normalized[split:]
    if len(val_data) == 0:
        raise ValueError(f"Insufficient data for {country_name} to hold out an ARIMA validation split.")

    model = ARIMA(train_data, order=ARIMA_ORDER)
    fitted = model.fit()

    Path(save_path).mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, f"{save_path}/{country_name}_arima_model.pkl")

    val_forecast = fitted.forecast(steps=len(val_data))
    mae = mean_absolute_error(val_data, val_forecast)
    mse = mean_squared_error(val_data, val_forecast)
    r2 = r2_score(val_data, val_forecast)

    max_year = long_format[long_format["Country"] == country_name]["Year"].max()
    steps = target_year - max_year
    if steps <= 0:
        raise ValueError(f"target_year {target_year} is already within {country_name}'s training data.")
    full_forecast = fitted.forecast(steps=steps)
    prediction = scaler.inverse_transform([[full_forecast[-1]]]).flatten()[0]

    return ModelResult("ARIMA", mae, float(prediction), mse, float(np.sqrt(mse)), r2)


def train_sarima(country_name: str, long_format: pd.DataFrame, target_year: int,
                  order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                  save_path: str = "sarima_models") -> ModelResult:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler

    series = get_country_series(long_format, country_name)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    split = int(len(normalized) * 0.8)
    train_data, val_data = normalized[:split], normalized[split:]
    if len(val_data) == 0:
        raise ValueError(f"Insufficient data for {country_name} to hold out a SARIMA validation split.")

    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, f"{save_path}/{country_name}_sarima_model.pkl")

    val_forecast = fitted.get_forecast(steps=len(val_data)).predicted_mean
    mae = mean_absolute_error(val_data, val_forecast)
    mse = mean_squared_error(val_data, val_forecast)
    r2 = r2_score(val_data, val_forecast)

    max_year = long_format[long_format["Country"] == country_name]["Year"].max()
    steps = target_year - max_year
    if steps <= 0:
        raise ValueError(f"target_year {target_year} is already within {country_name}'s training data.")
    full_forecast = fitted.get_forecast(steps=steps).predicted_mean
    prediction = scaler.inverse_transform([[full_forecast[-1]]]).flatten()[0]

    return ModelResult("SARIMA", mae, float(prediction), mse, float(np.sqrt(mse)), r2)


def train_prophet(country_name: str, long_format: pd.DataFrame, target_year: int,
                   save_path: str = "prophet_models") -> ModelResult:
    from prophet import Prophet
    from sklearn.preprocessing import MinMaxScaler

    country_data = long_format[long_format["Country"] == country_name].sort_values("Year")
    if len(country_data) < 2:
        raise ValueError(f"Insufficient data for {country_name}; Prophet needs >= 2 points.")

    prophet_data = country_data[["Year", "Emission"]].rename(columns={"Year": "ds", "Emission": "y"})
    prophet_data["ds"] = pd.to_datetime(prophet_data["ds"], format="%Y")

    scaler = MinMaxScaler()
    prophet_data["y"] = scaler.fit_transform(prophet_data[["y"]])

    split = int(len(prophet_data) * 0.8)
    train_data, val_data = prophet_data.iloc[:split], prophet_data.iloc[split:]

    model = Prophet()
    model.fit(train_data)

    future = model.make_future_dataframe(periods=len(val_data), freq="Y")
    forecast = model.predict(future)
    val_forecast = forecast[-len(val_data):]

    mae = mean_absolute_error(val_data["y"], val_forecast["yhat"])
    mse = mean_squared_error(val_data["y"], val_forecast["yhat"])
    r2 = r2_score(val_data["y"], val_forecast["yhat"])

    Path(save_path).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{save_path}/{country_name}_prophet_model.pkl")

    max_year = int(prophet_data["ds"].dt.year.max())
    periods = target_year - max_year
    if periods <= 0:
        raise ValueError(f"target_year {target_year} is already within {country_name}'s training data.")
    full_future = model.make_future_dataframe(periods=periods, freq="Y")
    full_forecast = model.predict(full_future)
    prediction = scaler.inverse_transform([[full_forecast["yhat"].iloc[-1]]]).flatten()[0]

    return ModelResult("Prophet", mae, float(prediction), mse, float(np.sqrt(mse)), r2)


# --------------------------------------------------------------------------- #
# Recurrent neural models — TensorFlow imported lazily, only when called
# --------------------------------------------------------------------------- #

def _build_rnn(cell_type: str, input_timesteps: int):
    """cell_type in {'lstm', 'bilstm', 'gru'}; matches notebook architecture exactly."""
    import tensorflow as tf  # lazy import — keeps classical models TF-free
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout

    if cell_type == "lstm":
        recurrent_layer = LSTM(RNN_UNITS, activation="relu", input_shape=(input_timesteps, 1),
                                return_sequences=False)
        layers = [recurrent_layer]
    elif cell_type == "bilstm":
        layers = [Bidirectional(LSTM(RNN_UNITS, activation="relu", return_sequences=False),
                                 input_shape=(input_timesteps, 1))]
    elif cell_type == "gru":
        layers = [GRU(RNN_UNITS, activation="relu", input_shape=(input_timesteps, 1),
                       return_sequences=False)]
    else:
        raise ValueError(f"Unknown cell_type '{cell_type}'.")

    model = Sequential(layers + [
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _train_recurrent(cell_type: str, country_name: str, long_format: pd.DataFrame,
                      target_year: int) -> ModelResult:
    series = get_country_series(long_format, country_name)
    dataset = build_windowed_dataset(series, sequence_length=SEQUENCE_LENGTH)
    X_train, X_val, X_test = dataset.reshaped_for_rnn()

    model = _build_rnn(cell_type, input_timesteps=SEQUENCE_LENGTH)
    history = model.fit(
        X_train, dataset.y_train,
        validation_data=(X_val, dataset.y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
    )

    test_loss, test_mae = model.evaluate(X_test, dataset.y_test, verbose=0)
    test_preds = model.predict(X_test, verbose=0)
    mse = mean_squared_error(dataset.y_test, test_preds)
    r2 = r2_score(dataset.y_test, test_preds)

    max_year = int(long_format[long_format["Country"] == country_name]["Year"].max())
    prediction = recursive_forecast(model, dataset.scaler, series, max_year, target_year,
                                     sequence_length=SEQUENCE_LENGTH)

    display_name = {"lstm": "LSTM", "bilstm": "Bi-LSTM", "gru": "GRU"}[cell_type]
    return ModelResult(
        display_name, float(test_mae), prediction, float(mse), float(np.sqrt(mse)), float(r2),
        extra={"train_loss": history.history["loss"], "val_loss": history.history["val_loss"]},
    )


def train_lstm(country_name, long_format, target_year):
    return _train_recurrent("lstm", country_name, long_format, target_year)


def train_bilstm(country_name, long_format, target_year):
    return _train_recurrent("bilstm", country_name, long_format, target_year)


def train_gru(country_name, long_format, target_year):
    return _train_recurrent("gru", country_name, long_format, target_year)


# --------------------------------------------------------------------------- #
# Orchestration: train all six, select best by MAE (per README/thesis rule)
# --------------------------------------------------------------------------- #

MODEL_TRAINERS = {
    "ARIMA": train_arima,
    "SARIMA": train_sarima,
    "Prophet": train_prophet,
    "LSTM": train_lstm,
    "Bi-LSTM": train_bilstm,
    "GRU": train_gru,
}


def train_all_models(country_name: str, long_format: pd.DataFrame, target_year: int,
                      models: Optional[list] = None) -> dict:
    """
    Train every requested model (default: all six) for one country and return
    a {model_name: ModelResult} dict. Failures on any single model are caught
    and recorded rather than aborting the whole run, since sparse series
    (e.g. small countries with 15-20 years of data) can legitimately fail one
    architecture (typically SARIMA/ARIMA) while others succeed.
    """
    models = models or list(MODEL_TRAINERS.keys())
    results = {}
    for name in models:
        trainer = MODEL_TRAINERS[name]
        try:
            results[name] = trainer(country_name, long_format, target_year)
        except Exception as exc:  # noqa: BLE001 — surfaced to caller, not silenced
            results[name] = ModelResult(name, mae=float("inf"), prediction=float("nan"),
                                         extra={"error": str(exc)})
    return results


def select_best_model(results: dict) -> ModelResult:
    """Least-MAE selection rule, matching thesis §6.4 / §7.2 exactly."""
    valid = {k: v for k, v in results.items() if np.isfinite(v.mae)}
    if not valid:
        raise RuntimeError("No model trained successfully for this country/year.")
    best_name = min(valid, key=lambda k: valid[k].mae)
    return valid[best_name]
