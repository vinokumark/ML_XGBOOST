import os
import re
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from transformers import pipeline

# ------------------
# Config / Cache Files
# ------------------
MODEL_FILE = "loan_model.pkl"
CONFIG_FILE = "loan_config.pkl"

# sLM (small language model)
llm = pipeline("text2text-generation", model="google/flan-t5-small")

# ------------------
# Load Loan Dataset with PII
# ------------------
def load_dataset():
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv")
    df = df.rename(columns=str.lower)

    # add dummy PII fields
    df["customer_name"] = [f"Customer_{i}" for i in range(len(df))]
    df["email"] = [f"user{i}@mail.com" for i in range(len(df))]
    df["phone"] = [f"+91-900000{i:04d}" for i in range(len(df))]

    return df

df = load_dataset()

# ------------------
# Detect Target Column
# ------------------
def detect_target_column(df):
    for col in df.columns:
        if col.lower() in ["class", "creditability", "target", "label", "credit_risk"]:
            return col
    return None

TARGET_COL = detect_target_column(df)
if TARGET_COL is None:
    st.error(f"❌ No target column found! Available columns: {list(df.columns)}")
    st.stop()
else:
    st.info(f"✅ Detected target column: **{TARGET_COL}** | Values: {df[TARGET_COL].unique().tolist()}")

# ------------------
# Tokenization (mask PII)
# ------------------
def tokenize_pii(df):
    df_tok = df.copy()
    if "customer_name" in df_tok.columns:
        df_tok["customer_name"] = "NAME_MASK"
    if "email" in df_tok.columns:
        df_tok["email"] = "EMAIL_MASK"
    if "phone" in df_tok.columns:
        df_tok["phone"] = "PHONE_MASK"
    return df_tok

# ------------------
# Preprocessing
# ------------------
def preprocess_data(use_tokenization):
    data = tokenize_pii(df) if use_tokenization else df.copy()

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    # convert categorical → numeric
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target if it's categorical (e.g. good/bad → 0/1)
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y

# ------------------
# Train & Evaluate
# ------------------
def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc

# ------------------
# Streamlit UI
# ------------------
st.set_page_config(layout="wide")
st.title("🏦 Loan Approval Prediction System")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📂 Dataset", "⚡ Training & Accuracy", "📊 Feature Importance", "🔎 Query + Explainability"]
)

# ------------------
# Tab 1: Dataset
# ------------------
with tab1:
    st.subheader("📂 Dataset Preview")
    st.write(df.head())

# ------------------
# Tab 2: Training & Accuracy
# ------------------
with tab2:
    st.subheader("⚡ Train Models")

    token_switch = st.radio("Enable Tokenization?", ["ON", "OFF"])
    use_tokenization = token_switch == "ON"

    if st.button("Train Model"):
        X, y = preprocess_data(use_tokenization)
        model, acc = train_and_eval(X, y)

        joblib.dump(model, MODEL_FILE)
        joblib.dump({"use_tokenization": use_tokenization, "feature_names": list(X.columns)}, CONFIG_FILE)

        st.success(f"Model trained ✅ | Accuracy: {acc:.2f}")

# ------------------
# Tab 3: Feature Importance
# ------------------
with tab3:
    st.subheader("📊 Feature Importance")
    if os.path.exists(MODEL_FILE) and os.path.exists(CONFIG_FILE):
        model = joblib.load(MODEL_FILE)
        cfg = joblib.load(CONFIG_FILE)
        use_tokenization = cfg["use_tokenization"]
        X, _ = preprocess_data(use_tokenization)
        importance = model.feature_importances_

        fi = pd.DataFrame({"feature": X.columns, "importance": importance}).sort_values(by="importance", ascending=False)
        st.dataframe(fi)
    else:
        st.warning("⚠️ Train a model in Tab 2 first (no saved model found).")

# ------------------
# Tab 4: Query + Explainability
# ------------------
with tab4:
    st.subheader("🔎 Query + Explainability")

    if os.path.exists(MODEL_FILE) and os.path.exists(CONFIG_FILE):
        cfg = joblib.load(CONFIG_FILE)
        use_tokenization_active = cfg["use_tokenization"]
        trained_features = cfg["feature_names"]
        st.info(f"Active model: **{'Tokenization ON' if use_tokenization_active else 'Tokenization OFF'}**")
    else:
        st.warning("⚠️ No active model found. Train in Tab 2 first.")
        st.stop()

    cust_idx = st.number_input("Enter customer row index:", 0, len(df)-1, 0)
    if st.button("Check Loan Eligibility"):
        model = joblib.load(MODEL_FILE)
        X_active, _ = preprocess_data(use_tokenization_active)
        customer = X_active.iloc[[cust_idx]].reindex(columns=trained_features, fill_value=0)
        pred = model.predict(customer)[0]
        decision = "❌ NO (Not Eligible)" if pred == 1 else "✅ YES (Eligible)"
        st.write(f"Row {cust_idx}: {decision}")

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(customer)
        st.subheader("📊 SHAP Explanation")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    # ------------------
    # Natural Language Query
    # ------------------
    st.write("---")
    st.subheader("💬 Natural Language Query")

    query = st.text_input("Ask (e.g., 'show customers below age 30', 'show customers not eligible')")

    if query:
        prompt = f"""
Convert this into Pandas code.
Rules:
- Use 'df' for dataset.
- Use 'X_active' for processed features.
- Use 'model.predict(X_active)' for eligibility.
- Return ONLY valid Python code.

Question: {query}
Code:
"""
        response = llm(prompt, max_length=128)[0]["generated_text"].strip()
        fallback_code = None

        # --- fallback regex ---
        if "not eligible" in query.lower():
            fallback_code = "df.iloc[X_active[model.predict(X_active)==1].index]"
        elif "eligible" in query.lower() and "not" not in query.lower():
            fallback_code = "df.iloc[X_active[model.predict(X_active)==0].index]"
        elif "age" in query.lower():
            nums = re.findall(r"\d+", query)
            if nums:
                if "below" in query.lower():
                    fallback_code = f"df[df['age'] < {nums[0]}]"
                elif "above" in query.lower():
                    fallback_code = f"df[df['age'] > {nums[0]}]"
        elif "amount" in query.lower() or "credit_amount" in query.lower():
            nums = re.findall(r"\d+", query)
            if nums:
                if "below" in query.lower():
                    fallback_code = f"df[df['amount'] < {nums[0]}]"
                elif "above" in query.lower():
                    fallback_code = f"df[df['amount'] > {nums[0]}]"
        elif "duration" in query.lower():
            nums = re.findall(r"\d+", query)
            if nums:
                if "below" in query.lower():
                    fallback_code = f"df[df['duration'] < {nums[0]}]"
                elif "above" in query.lower():
                    fallback_code = f"df[df['duration'] > {nums[0]}]"

        # pick response or fallback
        if not response or "df" not in response:
            response = fallback_code

        st.write("**Raw Query**")
        st.write(query)
        st.write("**Generated Code**")
        st.code(response, language="python")

        # run
        try:
            model = joblib.load(MODEL_FILE)
            X_active, _ = preprocess_data(use_tokenization_active)
            X_active = X_active.reindex(columns=trained_features, fill_value=0)
            safe_locals = {"df": df, "X_active": X_active, "model": model, "pd": pd}
            result = eval(response, {"__builtins__": {}}, safe_locals)
            st.write("**Execution Result**")
            if isinstance(result, (int, float)):
                decision = "❌ NO (Not Eligible)" if int(result) == 1 else "✅ YES (Eligible)"
                st.success(f"Answer: {decision}")
            else:
                st.dataframe(result)
        except Exception as e:
            st.error(f"Execution failed: {e}")

