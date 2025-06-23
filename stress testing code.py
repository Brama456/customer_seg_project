import streamlit as st
import pandas as pd
import numpy as np
import time
import psutil
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Page config
st.set_page_config(page_title="Stress Test: Banking Clustering", layout="wide")
st.title("🧪 Stress Testing - Banking Customer Clustering with Memory Monitoring")

# Memory usage checker
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    return mem_bytes / (1024 ** 2)  # in MB

# Load data
DEFAULT_PATH = r"C:\Users\Bramarambika\Downloads\Workoopolis\Customer_segmentation\customer_segmentation_banking.xlsx"
if os.path.exists(DEFAULT_PATH):
    df = pd.read_excel(DEFAULT_PATH)
else:
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        st.stop()

st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Encode categorical features
for col in ['gender', 'marital_status', 'account_type']:
    df[col] = df[col].fillna("Unknown")
    df[col] = LabelEncoder().fit_transform(df[col])

# Add age group feature
df['age_group'] = pd.cut(df['age'], bins=[0, 35, 55, 100], labels=['Young', 'Mid', 'Senior'])
df['age_group'] = LabelEncoder().fit_transform(df['age_group'].astype(str))

# Feature selection
features = [
    'age', 'income', 'recency', 'monetary',
    'digital_channel_usage_score', 'total_transactions',
    'average_transaction_value', 'loan_products', 'investment_products',
    'gender', 'marital_status', 'account_type', 'age_group'
]

# Clean and scale data
X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
df = df.loc[X.index]

# --- Preprocessing ---
st.info("⚙️ Preprocessing data...")
mem_before_pre = get_memory_usage()
start_pre = time.time()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

end_pre = time.time()
mem_after_pre = get_memory_usage()
st.success(f"✅ Preprocessing completed in {end_pre - start_pre:.2f} seconds")
st.info(f"🧠 Memory used during preprocessing: {mem_after_pre - mem_before_pre:.2f} MB")

# --- Clustering ---
st.info("📊 Running KMeans clustering...")
mem_before_model = get_memory_usage()
start_model = time.time()

try:
    kmeans = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    end_model = time.time()
    mem_after_model = get_memory_usage()
    st.success(f"✅ Clustering completed in {end_model - start_model:.2f} seconds")
    st.info(f"🧠 Memory used during clustering: {mem_after_model - mem_before_model:.2f} MB")
except Exception as e:
    st.error(f"❌ Clustering failed: {e}")
    st.stop()

# --- Segmentation Output ---
st.subheader("📌 Sample of Segmented Data (Top 100 Rows)")
st.dataframe(df.head(100))  # Limit display to avoid lag

# --- Download segmented data ---
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Full Segmented Data", csv, "segmented_customers.csv", "text/csv")

# --- Cluster Profiling ---
st.subheader("📊 Cluster Profiling Summary")
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
