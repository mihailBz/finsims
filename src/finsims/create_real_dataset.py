import argparse

import yfinance as yf
from jsonl_utils import save_dataset
from transformations import log_return


def main(format_, data_dir, n, symbols):
    for symbol in symbols:
        data = yf.download(symbol, period="max")
        data = data["Adj Close"]
        log_returns = log_return(data.dropna().to_numpy())
        max_length = log_returns.shape[0] - log_returns.shape[0] % n
        log_returns = log_returns[:max_length].reshape(-1, n)
        save_dataset(f"{data_dir}/{symbol}_log_return", log_returns, format_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a dataset of log returns from real stock data."
    )
    parser.add_argument(
        "-f",
        "--format",
        default="tsdiff",
        type=str,
        help="The format to save the dataset (e.g., 'diffusionts' or 'tsdiff').",
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="data",
        help="The directory to save the datasets.",
    )

    parser.add_argument(
        "-n",
        type=int,
        default=200,
        help="The number of time steps.",
    )

    parser.add_argument(
        "-s",
        "--symbols",
        nargs="+",
        default=[],
        help="The symbols to download.",
    )

    args = parser.parse_args()

    main(args.format, args.data_dir, args.n, args.symbols)
