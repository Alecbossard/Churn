import streamlit as st
import pandas as pd

from churn_platform.data.loader import load_processed_split
from churn_platform.models.persistence import load_model


@st.cache_data
def load_data():
    """
    Load train, val, test splits and combine them.
    """
    train_df = load_processed_split("train")
    val_df = load_processed_split("val")
    test_df = load_processed_split("test")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return full_df, test_df


@st.cache_resource
def load_churn_model():
    """
    Load the final churn model from disk.
    By default we use the tuned random forest.
    """
    model = load_model(filename="rf_tuned.joblib")
    return model


def main() -> None:
    st.set_page_config(
        page_title="Churn Analytics Dashboard",
        layout="wide",
    )

    st.title("Churn Analytics Dashboard")

    full_df, test_df = load_data()
    model = load_churn_model()

    # Basic sanity: make sure target exists
    if "ChurnFlag" not in full_df.columns:
        st.error("ChurnFlag column not found in data.")
        return

    tab_overview, tab_segments, tab_scoring = st.tabs(
        ["Overview", "Segments", "Customer scoring"]
    )

    # -----------------------
    # Tab 1: Overview
    # -----------------------
    with tab_overview:
        st.subheader("Global KPIs")

        churn_rate = full_df["ChurnFlag"].mean()
        n_customers = len(full_df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Global churn rate",
                f"{churn_rate * 100:.1f}%",
            )
        with col2:
            st.metric(
                "Number of customers",
                f"{n_customers}",
            )

        st.markdown("---")

        c1, c2 = st.columns(2)

        # Churn by Contract
        with c1:
            st.subheader("Churn rate by contract type")

            if "Contract" in full_df.columns:
                contract_stats = (
                    full_df.groupby("Contract")["ChurnFlag"]
                    .mean()
                    .reset_index()
                    .rename(columns={"ChurnFlag": "churn_rate"})
                )
                contract_stats["churn_rate_percent"] = contract_stats["churn_rate"] * 100

                st.dataframe(contract_stats, use_container_width=True)

                st.bar_chart(
                    contract_stats.set_index("Contract")["churn_rate_percent"]
                )
            else:
                st.info("Contract column not found in data.")

        # Tenure distribution
        with c2:
            st.subheader("Tenure distribution")

            if "tenure" in full_df.columns:
                st.caption("Histogram of customer tenure (months).")

                # Build a small histogram with ~20 bins
                hist_data = full_df["tenure"].value_counts(
                    bins=20, sort=False
                ).reset_index()
                hist_data.columns = ["tenure_bin", "count"]
                hist_data["tenure_bin"] = hist_data["tenure_bin"].astype(str)

                st.bar_chart(hist_data.set_index("tenure_bin")["count"])
            else:
                st.info("tenure column not found in data.")

    # -----------------------
    # Tab 2: Segments
    # -----------------------
    with tab_segments:
        st.subheader("Churn by segment")

        # Side controls
        segment_type = st.selectbox(
            "Segment by:",
            ["Contract", "tenure (binned)", "TotalCharges (binned)"],
        )

        df = full_df.copy()

        if segment_type == "Contract":
            if "Contract" not in df.columns:
                st.warning("Contract column not found.")
            else:
                stats = (
                    df.groupby("Contract")["ChurnFlag"]
                    .agg(count="size", churn_rate="mean")
                    .reset_index()
                )
                stats["churn_rate_percent"] = stats["churn_rate"] * 100
                st.dataframe(stats, use_container_width=True)
                st.bar_chart(stats.set_index("Contract")["churn_rate_percent"])

        elif segment_type == "tenure (binned)":
            if "tenure" not in df.columns:
                st.warning("tenure column not found.")
            else:
                df["tenure_bin"] = pd.qcut(
                    df["tenure"], q=5, duplicates="drop"
                )
                stats = (
                    df.groupby("tenure_bin")["ChurnFlag"]
                    .agg(count="size", churn_rate="mean")
                    .reset_index()
                )
                stats["churn_rate_percent"] = stats["churn_rate"] * 100
                st.dataframe(stats, use_container_width=True)
                st.bar_chart(
                    stats.set_index("tenure_bin")["churn_rate_percent"]
                )

        elif segment_type == "TotalCharges (binned)":
            if "TotalCharges" not in df.columns:
                st.warning("TotalCharges column not found.")
            else:
                df["TotalCharges_bin"] = pd.qcut(
                    df["TotalCharges"], q=5, duplicates="drop"
                )
                stats = (
                    df.groupby("TotalCharges_bin")["ChurnFlag"]
                    .agg(count="size", churn_rate="mean")
                    .reset_index()
                )
                stats["churn_rate_percent"] = stats["churn_rate"] * 100
                st.dataframe(stats, use_container_width=True)
                st.bar_chart(
                    stats.set_index("TotalCharges_bin")["churn_rate_percent"]
                )

    # -----------------------
    # Tab 3: Customer scoring
    # -----------------------
    with tab_scoring:
        st.subheader("Score an existing customer (from test set)")

        if "customerID" not in test_df.columns:
            st.warning("customerID column not found in test data.")
        else:
            customer_ids = test_df["customerID"].tolist()
            selected_id = st.selectbox("Select customerID", customer_ids)

            row = test_df[test_df["customerID"] == selected_id].iloc[0]
            st.write("Raw customer data:")
            st.json(row.to_dict())

            # Prepare a 1-row DataFrame for prediction
            input_df = row.to_frame().T.copy()

            # Drop target columns if present
            for col in ["ChurnFlag", "Churn"]:
                if col in input_df.columns:
                    input_df = input_df.drop(columns=[col])

            proba = model.predict_proba(input_df)[0, 1]

            true_label = row.get("ChurnFlag", None)

            st.markdown("### Prediction")
            col_left, col_right = st.columns(2)

            with col_left:
                st.metric(
                    "Predicted churn probability",
                    f"{proba * 100:.1f}%",
                )

            with col_right:
                threshold = st.slider(
                    "Decision threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.5,
                    step=0.05,
                )
                predicted_label = 1 if proba >= threshold else 0
                label_str = "Churn" if predicted_label == 1 else "No churn"

                if predicted_label == 1:
                    st.error(f"Predicted label (threshold {threshold:.2f}): **{label_str}**")
                else:
                    st.success(f"Predicted label (threshold {threshold:.2f}): **{label_str}**")

            if true_label is not None:
                true_str = "Churn" if true_label == 1 else "No churn"
                st.write(f"True label: **{true_str}**")

            st.info(
                "This scoring uses the same tuned random forest pipeline as in the Python scripts "
                "(including preprocessing and one-hot encoding)."
            )


if __name__ == "__main__":
    main()
