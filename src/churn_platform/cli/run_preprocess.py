from churn_platform.data.loader import load_raw_data
from churn_platform.data.preprocess import clean_telco_data, save_interim_data
from churn_platform.data.split import train_val_test_split, save_splits


def main() -> None:
    print("Loading raw data...")
    raw_df = load_raw_data()
    print(f"Raw shape: {raw_df.shape}")

    print("Cleaning data...")
    clean_df = clean_telco_data(raw_df)
    print(f"Cleaned shape: {clean_df.shape}")

    print("Saving interim cleaned data...")
    interim_path = save_interim_data(clean_df)
    print(f"Interim data saved to: {interim_path}")

    print("Splitting into train / val / test...")
    train_df, val_df, test_df = train_val_test_split(clean_df)

    print("Saving train / val / test splits...")
    save_splits(train_df, val_df, test_df)

    print("Preprocessing and splitting finished.")


if __name__ == "__main__":
    main()
