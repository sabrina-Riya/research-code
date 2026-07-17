# CO2 Emissions Forecasting & Carbon Storage Optimization

Predictive modeling of country-level CO2 emissions using classical time-series
methods (ARIMA, SARIMA, Prophet) and recurrent neural networks (LSTM, Bi-LSTM,
GRU), coupled with a geospatial/economic optimization layer (Linear scoring +
TOPSIS/WSM/VIKOR) for CO2 storage site selection.

## 1. Forecasting Approach

All three neural models (LSTM, Bi-LSTM, GRU) are trained on 3-year sliding
windows of historical emissions (1990–2018) and forecast forward to a target
year (e.g. 2026) using **recursive (autoregressive) multi-step forecasting**:
each model predicts one year ahead, appends that prediction to the input
sequence, and slides the window forward to predict the next year. This repeats
until the target year is reached (an 8-step horizon for a 2026 target).

This is a deliberate design choice over direct multi-step forecasting, because
it lets the model condition each step on the most recent trend — including its
own most recent prediction — rather than committing to a single long-range
extrapolation from 2018 data alone.

## 2. The Bottleneck: Compounding Forecast Error

Recursive forecasting has a well-known structural weakness: once a predicted
value re-enters the input window, any error in that value is no longer just
an isolated output — it becomes part of the *context* the model conditions on
for every subsequent step. As the recursion continues, that error is
reprocessed through the recurrent gates repeatedly and can grow or shrink
depending on the local sensitivity (Jacobian) of the model at each step. This
is the forward-pass analogue of the vanishing/exploding gradient problem seen
in backpropagation through time: instead of a gradient being propagated
backward through the unrolled network, a *prediction error* is propagated
forward through it, and the sensitivity of the final (2026) forecast to an
early error compounds multiplicatively across the recursion length.

In practice this shows up exactly where you'd expect:

- **Bangladesh** (smooth, near-linear historical trend) → low first-step
  error → little for the recursion to amplify → tight, stable long-horizon
  forecasts.
- **Sweden** (volatile, non-monotonic trend with sharp drops/rebounds) →
  larger first-step error and higher local sensitivity → more room for the
  recursive loop to compound deviation over 8 steps.

This is a real limitation of the pipeline, and it isn't solved by an explicit
error-correction term inside the recursive loop itself — no such term exists
in this implementation. Instead, the impact of compounding error is kept in
check indirectly, through choices made upstream of and around the recursion
rather than inside it.

## 3. How It's Kept in Check (Mitigating, Not Eliminating)

The pipeline does **not** implement a targeted correction for recursive error
amplification. What limits its practical impact are three structural
decisions:

1. **Bidirectional context at the base-model level (Bi-LSTM).**
   Because Bi-LSTM's forward and backward passes are combined during
   training, its single-step predictions are lower-variance and better
   calibrated than a unidirectional model's. Lower per-step error going *into*
   the recursion means less for the recursion to compound — this is why
   Bi-LSTM consistently posts the lowest MAE/RMSE across all three test
   countries and was selected as the best-performing model overall.

2. **A short forecasting horizon.**
   The recursion only runs for as many steps as there are years between the
   last observed data point (2018) and the target year (e.g. 8 steps to
   2026). Compounding error scales with the number of multiplicative
   amplification steps, so keeping the horizon short is itself a form of
   damage control — it caps how many times an error can be re-processed.

3. **Standard regularization during training (dropout, weight decay).**
   These constrain the learned weights to reduce overfitting and keep the
   per-step Jacobians well-behaved, but they act on the *training* process,
   not on the *inference-time* recursive loop — they reduce the odds of a bad
   first-step prediction, not the propagation of one once it happens.

**Net effect:** the pipeline reduces the *probability and magnitude* of
compounding error (via a stronger base model, a short horizon, and
regularized training) rather than *correcting* it once it occurs. This is
reflected in the results — Bi-LSTM's R² stays above 0.98 even on Sweden's
volatile series, while classical models with no equivalent architecture (and
no recursion) fail outright on non-stationary series (e.g. SARIMA R² of
-44.96 on Sweden) for unrelated reasons (rigid seasonality/stationarity
assumptions).

## 4. What a Full Fix Would Look Like (Not Implemented Here)

For completeness, options that would directly target recursive error
propagation rather than working around it include:

- **Scheduled sampling** during training (mixing ground-truth and
  model-predicted inputs so the model learns to tolerate its own errors).
- **Direct multi-step (seq2seq) forecasting** instead of recursion, trading
  the compounding-error problem for a harder single-shot prediction problem.
- **Uncertainty-aware recursion**, propagating a confidence interval forward
  alongside the point prediction and widening it at each step.
- **Explicit error-correction/residual models** trained on the historical
  divergence between recursive predictions and ground truth.

None of these are implemented in the current codebase; they are noted here as
the natural next step for anyone extending the horizon beyond ~8 years or
applying this pipeline to more volatile series than those tested.

## 5. Repository Structure

```
├── research_model_code.ipynb   # Data prep, model training, recursive forecasting,
│                                # MAE/MSE/RMSE/R² evaluation, site-selection logic
├── report_final.pdf            # Full thesis writeup (methodology, literature
│                                # review, results, Power BI visualizations)
└── README.md                   # This file
```

Within the notebook:

- **Preprocessing** — cleans/reshapes historical emissions (1990–2018) into
  `[year, emission]` sequences, Min-Max normalized, 80/20 train/test split.
- **Forecasting models** — `train_and_save_prophet_model`,
  `train_and_save_arima_model`, `train_and_save_sarima_model`, and the LSTM /
  Bi-LSTM / GRU training cells, each paired with a
  `predict_emission_recursive_*` function implementing the sliding-window
  recursion described above (`sequence_length=3`).
- **Model selection** — lowest-MAE model is chosen per country/year as the
  final emissions forecast.
- **Site selection** — Linear weighted-scoring method and three MCDA methods
  (TOPSIS, WSM, VIKOR) rank CCS storage sites by cost, distance, and capacity
  relative to the forecasted emissions.
