import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Feature columns (must match training)
# -----------------------------
cat_cols = ['loan_type', 'Neg_ammortization', 'loan_purpose']
num_cols = ['rate_of_interest','property_value','LTV','dtir1','income','loan_amount']

# -----------------------------
# Load trained pipeline
# -----------------------------
MODEL_PATH = "./data/GradientBoostingClassifier_LOANS_STATUS.pkl"
best_model = joblib.load(MODEL_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Loan Status Predictor", page_icon="💳", layout="wide")
st.title("💳 Loan Status Prediction Dashboard")

st.markdown("Upload a CSV file with loan applications to predict approval status.")

uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        predictions = best_model.predict(df)
        probabilities = best_model.predict_proba(df)

        df["Predicted_Status"] = predictions
        df["Prob_Not_Approved"] = probabilities[:,0].round(3)
        df["Prob_Approved"] = probabilities[:,1].round(3)

        st.subheader("✅ Prediction Results")
        st.dataframe(df)

        # Download results
        csv_output = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv_output,
            file_name="loan_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure your pipeline was trained and saved with the same scikit-learn version.")
