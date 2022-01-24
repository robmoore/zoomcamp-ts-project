import argparse
import logging
from typing import Tuple

import pandas as pd
import requests as requests
from numpy.random import default_rng

URL = "http://0.0.0.0:5000/predict"

logger = logging.getLogger(__name__)


def main(url: str) -> Tuple[float, float]:
    df = pd.read_pickle("bin/test.bin")

    rng = default_rng()
    index = rng.integers(len(df) - 26)

    historical_prices = df.iloc[index : index + 10]
    response = requests.post(url, json=historical_prices)
    logger.debug(f"request: {response.request.body}")
    logger.debug(f"response: {response.content}")

    return response.json()["prices"], df.iloc[index + 10 : index + 26].values


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
