from utils import build_dataframe


def main() -> None:
    df = build_dataframe()
    df.to_csv("data/gspc.csv")


if __name__ == "__main__":
    main()
