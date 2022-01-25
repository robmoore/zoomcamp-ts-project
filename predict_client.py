import argparse
from typing import List, Tuple

import requests as requests
from loguru import logger
from numpy.random import default_rng
import datetime as dt

from utils import build_dataframe

URL = "http://0.0.0.0:5000/predict"

label_columns = ["adjclose"]


def main(url: str) -> Tuple[List[float], List[float]]:
    df = build_dataframe(start="2020-01-01", end=dt.date.today().isoformat())

    df = df[label_columns]

    rng = default_rng()
    index = rng.integers(len(df) - 26)

    historical_prices = df.iloc[index : index + 16].squeeze().tolist()
    real_prices = df.iloc[index + 16 : index + 26].squeeze().tolist()

    response = requests.post(url, json=historical_prices)
    logger.debug(f"request: {response.request.body}")
    logger.debug(f"response: {response.content}")

    return response.json(), real_prices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-u",
        "--url",
        help="URL of service endpoint",
        type=str,
        default=URL,
    )

    args = parser.parse_args()

    predicted_prices, actual_prices = main(args.url)

    print(f"predicted prices: {predicted_prices}; actual prices: {actual_prices}")
