# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 17:32:56 2026

@author: BBarsch
"""

# app.py
import streamlit as st
import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

st.set_page_config(page_title="Explain My Regression", layout="wide")

st.title("Explain My Regression")
st.write("Upload a dataset, run a regression, and get a plain-English explanation of the results.")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

def make_narrative(result, X_cols, y_col, n_rows):
    r2 = float(result.rsquared)
    adj_r2 = float(result.rsquared_adj)

    # Pull coefficients + p-values
    params = result.params.drop("const", errors="ignore")
    pvals = result.pvalues.drop("const", errors="ignore")

    # Identify “strong” predictors (simple rule)
    sig = pvals[pvals < 0.05].sort_values()
    top = params.reindex(params.abs().sort_values(ascending=False).index)

    lines = []
    lines.append(f"### Plain-English Summary")
    lines.append(
        f"You ran a linear regression predicting **{y_col}** using **{len(X_cols)}** predictor(s) "
        f"on **{n_rows}** row(s)."
    )
    lines.append(f"**Model fit:** R² = **{r2:.3f}**, Adjusted R² = **{adj_r2:.3f}**.")

    if r2 < 0.2:
        lines.append("That R² is quite low, which usually means the predictors don’t explain much of the variation in the outcome (or the relationship isn’t very linear).")
    elif r2 < 0.5:
        lines.append("That R² suggests a moderate amount of the outcome’s variation is explained by your predictors.")
    else:
        lines.append("That R² is fairly strong, meaning your predictors explain a lot of the outcome’s variation (for a linear model).")

    # Significant predictors
    if len(sig) == 0:
        lines.append("**Statistical significance:** None of the predictors are statistically significant at the 5% level (p < 0.05).")
        lines.append("This can happen with small sample sizes, noisy data, or if the true relationship is weak/nonlinear.")
    else:
        lines.append(f"**Statistical significance:** {len(sig)} predictor(s) are significant at p < 0.05.")
        sig_list = ", ".join([f"`{c}` (p={sig[c]:.3g})" for c in sig.index[:6]])
        lines.append(f"Most significant predictors: {sig_list}" + (" ..." if len(sig) > 6 else ""))

    # Coefficient interpretation (top 3 by absolute effect)
    lines.append("### What the coefficients mean (holding other variables constant)")
    for col in top.index[:3]:
        coef = float(params[col])
        direction = "increases" if coef > 0 else "decreases"
        lines.append(
            f"- A 1-unit increase in **{col}** is associated with an estimated **{direction}** "
            f"of **{abs(coef):.4g}** in **{y_col}** on average."
        )

    # Practical cautions
    cautions = []
    if n_rows < 50:
        cautions.append("Your dataset is small (< 50 rows), so results may be unstable.")
    if len(X_cols) > max(1, n_rows // 10):
        cautions.append("You have many predictors relative to the number of rows, which can cause overfitting.")
    if len(cautions) > 0:
        lines.append("### Quick cautions")
        for c in cautions:
            lines.append(f"- {c}")

    lines.append("### Next steps you can try")
    lines.append("- Add interaction terms (e.g., `X1 * X2`) if you suspect combined effects.")
    lines.append("- Try a log transform if the target is skewed (e.g., `log(y)`).")
    lines.append("- Compare against Ridge/Lasso if you have many predictors.")
    return "\n".join(lines)

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    with st.sidebar:
        st.header("Model setup")
        y_col = st.selectbox("Choose target (y)", df.columns)
        X_cols = st.multiselect("Choose predictors (X)", [c for c in df.columns if c != y_col])

        missing_strategy = st.selectbox("Missing values", ["Drop rows with missing values", "Fill numeric with median"])
        test_size = st.slider("Test split size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random seed", 0, 9999, 42)

    if len(X_cols) == 0:
        st.warning("Select at least one predictor in the sidebar.")
        st.stop()

    data = df[[y_col] + X_cols].copy()

    # Handle missing values
    if missing_strategy == "Drop rows with missing values":
        data = data.dropna()
    else:
        # Fill numeric columns with median; leave categoricals to be handled via one-hot (which will drop NA rows if remain)
        for c in data.columns:
            if pd.api.types.is_numeric_dtype(data[c]):
                data[c] = data[c].fillna(data[c].median())
        data = data.dropna()

    # One-hot encode categoricals in X
    X_raw = data[X_cols]
    y = data[y_col]

    X = pd.get_dummies(X_raw, drop_first=True)
    # Ensure y numeric for OLS
    if not pd.api.types.is_numeric_dtype(y):
        st.error("Your target (y) must be numeric for linear regression. Choose a numeric column.")
        st.stop()

    n_rows = len(data)
    if n_rows < 10:
        st.error("Not enough rows after cleaning. Try a different missing-value strategy or dataset.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

    # Fit OLS with intercept
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm = sm.add_constant(X_test, has_constant="add")

    model = sm.OLS(y_train, X_train_sm).fit()

    # Predictions + metrics
    y_pred = model.predict(X_test_sm)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))


    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Metrics")
        st.metric("R² (train)", f"{model.rsquared:.3f}")
        st.metric("Adj. R² (train)", f"{model.rsquared_adj:.3f}")
        st.metric("MAE (test)", f"{mae:.3f}")
        st.metric("RMSE (test)", f"{rmse:.3f}")

        st.subheader("Coefficient Table")
        summary_df = pd.DataFrame({
            "coef": model.params,
            "p_value": model.pvalues
        }).sort_values("p_value")
        st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.subheader("Plots")

        # Predicted vs Actual
        fig1 = plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual")
        st.pyplot(fig1)

        # Residuals vs Fitted
        fitted = model.predict(X_train_sm)
        resid = y_train - fitted
        fig2 = plt.figure()
        plt.scatter(fitted, resid)
        plt.axhline(0)
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        st.pyplot(fig2)

    st.divider()
    st.markdown(make_narrative(model, list(X.columns), y_col, n_rows))

    with st.expander("Full statsmodels summary (optional)"):
        st.text(model.summary())
else:
    st.info("Upload a CSV to get started.")
