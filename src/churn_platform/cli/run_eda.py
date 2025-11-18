from churn_platform.data.loader import load_raw_data
from churn_platform.visualization.plots import (
    plot_target_distribution,
    plot_numeric_distributions,
    plot_categorical_vs_churn,
)


def main() -> None:
    print("Loading raw data...")
    data = load_raw_data()

    print("Data head:")
    print(data.head())

    print("\nDataFrame info:")
    # .info() prints to stdout, so we just call it
    data.info()

    print("\nPlotting target distribution...")
    plot_target_distribution(data)

    print("Plotting numeric distributions...")
    plot_numeric_distributions(data)

    print("Plotting some categorical variables vs Churn...")
    plot_categorical_vs_churn(data)


if __name__ == "__main__":
    main()
