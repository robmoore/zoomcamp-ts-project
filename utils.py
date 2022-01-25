from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from nbeats_keras.model import NBeatsNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tcn import TCN
from yahooquery import Ticker

from constants import INPUT_LENGTH, LABELS_LENGTH
from models import (
    last_baseline,
    lstm,
    nbeats,
    repeat_baseline,
    tcn,
    tcn_for_tuning,
    tuned_tcn,
)
from plotable_window_generator import PlotableWindowGenerator

MODEL_FUNCS = {
    "repeat": repeat_baseline,
    "last": last_baseline,
    "lstm": lstm,
    "tcn": tcn,
    "tuned_tcn": tuned_tcn,
    "nbeats": nbeats,
}


def split_dataset(data):
    train_df, test_df = train_test_split(data, test_size=0.2, shuffle=False)
    val_df, test_df = train_test_split(test_df, test_size=0.5, shuffle=False)

    return train_df, val_df, test_df


# evaluate one or more forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = []

    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        rmse = mean_squared_error(
            actual[:, : i + 1], predicted[:, : i + 1], squared=False
        )
        scores.append(rmse)

    score = sqrt(sum([s ** 2 for s in scores]) / len(scores))
    return score, scores


def summarize_scores(name, score, scores):
    day_scores = ", ".join([f"{i+1}: {s:.1f}" for i, s in enumerate(scores)])
    print(f"{name}: [{score:.3f}] {day_scores}")


def plot_scores(name, scores):
    days = [i + 1 for i in range(len(scores))]
    plt.plot(days, scores, marker="o", label=name)
    plt.show()


def build_model(
    model_name, window, verbose=0, epochs=1000, batch_size=32, patience=100
):
    model = MODEL_FUNCS[model_name]()

    compile_model(model)

    fit_model(model, window, verbose, epochs, batch_size, patience)

    return model


def fit_model(
    model,
    window,
    verbose=0,
    epochs=1000,
    batch_size=32,
    patience=100,
):
    model.fit(
        window.train(),
        validation_data=window.val(),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_root_mean_squared_error",
                min_delta=1e-4,
                patience=patience,
                verbose=1,
                mode="auto",
                restore_best_weights=True,
            ),
            tfa.callbacks.TQDMProgressBar(),
        ],
    )


def build_model_for_tuning(hp):
    model = tcn_for_tuning(hp)

    compile_model(model)

    return model


def compile_model(model):
    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()],
    )


# evaluate a single model
def evaluate_model(model, window, scaler):
    # walk-forward validation over each period
    actuals, predictions = generate_actuals_and_predictions(model, window, scaler)

    assert len(predictions) == len(actuals)

    score, scores = evaluate_forecasts(actuals, predictions)

    return score, scores


def build_dataframe(ticker_name="^GSPC", start="1999-01-01", end="2020-01-01"):
    ticker = Ticker(ticker_name)

    df = ticker.history(start=start, end=end)

    # Drop ticker in index and convert dates to datetimes
    df = df.loc[df.index.get_level_values(0)[0]]
    df.index = pd.to_datetime(df.index)

    return df


def build_window(df, label_columns):
    df = df[label_columns]

    # split into train and test
    train, val, test = split_dataset(df)

    # normalize data
    sc = StandardScaler()
    train.values[:] = sc.fit_transform(train)
    val.values[:], test.values[:] = sc.transform(val), sc.transform(test)

    return (
        PlotableWindowGenerator(
            train_df=train,
            val_df=val,
            test_df=test,
            input_width=INPUT_LENGTH,
            label_width=LABELS_LENGTH,
            shift=LABELS_LENGTH,
            label_columns=label_columns,
        ),
        sc,
    )


def generate_actuals_and_predictions(model, window, scaler):
    predictions = []
    actuals = []

    for inputs, labels in window.test():
        prediction = model.predict(inputs, verbose=0)
        predictions.extend(prediction)

        actual = labels.numpy()
        actual = actual.reshape(prediction.shape)
        actuals.extend(actual)

    predictions = np.array(predictions)
    predictions = np.squeeze(predictions)
    predictions = scaler.inverse_transform(predictions)

    actuals = np.array(actuals)
    actuals = np.squeeze(actuals)
    actuals = scaler.inverse_transform(actuals)

    return actuals, predictions


def plot_results(actuals, predictions, window, forecast_index=0, day_slice=slice(None)):
    offset_df = pd.DataFrame(
        {
            "actuals": actuals[:, forecast_index],
            "predictions": predictions[:, forecast_index],
        },
        index=window.test_df[
            INPUT_LENGTH + forecast_index - 1 : -LABELS_LENGTH + forecast_index
        ].index,
    )
    offset_df[day_slice].plot(figsize=(10, 6))


def load_model(model_name):
    return tf.keras.models.load_model(
        f"bin/{model_name}", custom_objects={"TCN": TCN, "NBeatsNet": NBeatsNet}
    )
