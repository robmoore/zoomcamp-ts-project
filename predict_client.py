import argparse
from typing import Tuple, List

import requests as requests
from numpy.random import default_rng
from loguru import logger

from utils import build_dataframe, split_dataset

URL = "http://0.0.0.0:5000/predict"

label_columns = ["adjclose"]


def main(url: str) -> Tuple[List[float], List[float]]:
    df = build_dataframe()

    df = df[label_columns]
    _, _, test = split_dataset(df)

    rng = default_rng()
    index = rng.integers(len(test) - 26)

    historical_prices = test.iloc[index : index + 16].squeeze().tolist()
    real_prices = test.iloc[index + 16 : index + 26].squeeze().tolist()

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
