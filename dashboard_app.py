
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

# Load data and model
df = pd.read_csv("cleaned_large_data.csv")
model = joblib.load("xgb_model.pkl")

# Feature Importances
st.title("Air Quality Prediction Dashboard")
st.write("## Feature Importances")
importances = model.feature_importances_
features = df.drop("AQI", axis=1).columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)

fig1, ax1 = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax1)
st.pyplot(fig1)

# SHAP summary plot
st.write("## SHAP Summary Plot")
explainer = shap.Explainer(model, df.drop("AQI", axis=1))
shap_values = explainer(df.drop("AQI", axis=1))
shap.plots.beeswarm(shap_values, show=False)
st.pyplot(bbox_inches='tight')
