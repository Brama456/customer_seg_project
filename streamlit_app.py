import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import os

# Page config
st.set_page_config(page_title="Banking Customer Segmentation", layout="wide")
st.title("📊 Banking Customer Segmentation")

# Load data
DEFAULT_PATH = r"C:\Users\Bramarambika\Downloads\Workoopolis\Customer_segmentation\customer_segmentation_banking.xlsx"
if os.path.exists(DEFAULT_PATH):
    df = pd.read_excel(DEFAULT_PATH)
else:
    st.warning("⚠️ Using cloud? Please upload your Excel file.")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        st.stop()

st.success("✅ Data loaded successfully")

# Encode categorical variables
for col in ['gender', 'marital_status', 'account_type']:
    df[col] = df[col].fillna("Unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Add age group as a feature
df['age_group'] = pd.cut(df['age'], bins=[0, 35, 55, 100], labels=['Young', 'Mid', 'Senior'])
df['age_group'] = LabelEncoder().fit_transform(df['age_group'].astype(str))

# Features used for clustering
features = [
    'age', 'income', 'recency', 'monetary',
    'digital_channel_usage_score', 'total_transactions',
    'average_transaction_value', 'loan_products', 'investment_products',
    'gender', 'marital_status', 'account_type', 'age_group'
]

# Clean and scale
X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
df = df.loc[X.index]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering with KMeans
kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Segmented data output
st.subheader("📌 Segmentation Results")
st.dataframe(df)

# Download segmented data
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

# Cluster profiling summary
st.subheader("📊 Cluster Profiling")
profile = df.groupby("cluster").agg({
    "age": "mean",
    "income": "mean",
    "recency": "mean",
    "monetary": "mean",
    "digital_channel_usage_score": "mean",
    "total_transactions": "mean",
    "average_transaction_value": "mean",
    "loan_products": "mean",
    "investment_products": "mean",
    "cluster": "count"
}).rename(columns={"cluster": "count"}).reset_index()

st.dataframe(profile)
