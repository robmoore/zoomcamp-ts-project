import pickle

from utils import build_dataframe, build_model, build_window

label_columns = ["adjclose"]
model_name = "tuned_tcn"


def main():
    # Read S&P 500 data
    df = build_dataframe()

    # Create a sliding window
    window, scaler = build_window(df, label_columns)

    # Generate model
    model = build_model(model_name, window)

    # Save model off in h5 format
    model.save(f"bin/{model_name}")

    # Save scaler for use in processing requests
    with open("bin/scaler.bin", "wb") as scaler_bin:
        pickle.dump(scaler, scaler_bin)

    # save for use in test client to randomly create request input
    window.test_df.to_pickle("bin/test.bin")


if __name__ == "__main__":
    main()
