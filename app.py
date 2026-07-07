# -*- coding: utf-8 -*-
"""
K-Means Clustering App with Dynamic UI and GitHub Data Source
"""

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# --- CONFIGURATION & TITLE ---
st.set_page_config(page_title="K-Means Clustering Lab", layout="wide")
st.title("📊 Clustering With K-Means - Interactive Tutorial")

# --- DYNAMIC DATA LOADING ---
# Pulling data dynamically from the provided GitHub repository url
DATA_URL = "https://raw.githubusercontent.com/Avinashonline160/KMEANS/main/income.csv"

@st.cache_data
def load_data(url):
    try:
        # Attempts to read directly from the GitHub raw link
        return pd.read_csv(url)
    except Exception:
        # Fallback dataset creation if the network/URL is unreachable or private
        st.warning("Could not connect to GitHub URL. Using placeholder data for demonstration.")
        data = {
            'Name': ['Rob', 'Michael', 'Mohan', 'Ismail', 'Kory', 'Gautam', 'David', 'Andrea', 'Brad', 'Angelina', 'Donald', 'Tom', 'Arnold', 'Jared', 'Stark', 'Ranbir', 'Dipika', 'Priyanka', 'Nick', 'Alia', 'Sid', 'Abdul'],
            'Age': [27, 29, 29, 28, 42, 39, 41, 38, 36, 35, 37, 26, 27, 28, 29, 49, 34, 45, 43, 31, 22, 23],
            'Income($)': [70000, 90000, 61000, 60000, 150000, 155000, 160000, 162000, 156000, 130000, 137000, 45000, 48000, 51000, 49500, 53000, 65000, 63000, 64000, 80000, 82000, 58000]
        }
        return pd.DataFrame(data)

df = load_data(DATA_URL)

# --- SIDEBAR UI CONTROLS ---
st.sidebar.header("🔧 Settings & Hyperparameters")
st.sidebar.markdown(f"[View GitHub Repository](https://github.com/Avinashonline160/KMEANS)")

# Dynamic cluster selection
k_value = st.sidebar.slider("Select Number of Clusters (K)", min_value=1, max_value=6, value=3, step=1)

# Feature Scaling toggle
apply_scaling = st.sidebar.checkbox("Apply MinMaxScaler Preprocessing", value=True)

# --- PREPROCESSING ---
df_processed = df.copy()

if apply_scaling:
    scaler = MinMaxScaler()
    df_processed['Income($)'] = scaler.fit_transform(df_processed[['Income($)']])
    df_processed['Age'] = scaler.fit_transform(df_processed[['Age']])

# --- K-MEANS MODELING ---
km = KMeans(n_clusters=k_value, random_state=42, n_init='auto')
y_predicted = km.fit_predict(df_processed[['Age', 'Income($)']])
df_processed['cluster'] = y_predicted

# --- USER INTERFACE LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df_processed.head(10), use_container_width=True)
    
    # --- ELBOW PLOT ---
    st.subheader("📈 The Elbow Method Plot")
    sse = []
    k_rng = range(1, 10)
    for k in k_rng:
        km_elbow = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km_elbow.fit(df_processed[['Age', 'Income($)']])
        sse.append(km_elbow.inertia_)
        
    fig_elbow, ax_elbow = plt.subplots(figsize=(5, 3.5))
    ax_elbow.set_xlabel('K Value')
    ax_elbow.set_ylabel('Sum of Squared Error (Inertia)')
    ax_elbow.plot(k_rng, sse, marker='o', color='purple')
    st.pyplot(fig_elbow)

with col2:
    st.subheader("🎯 K-Means Clusters Visualization")
    
    fig_cluster, ax_cluster = plt.subplots(figsize=(6, 5))
    colors = ['green', 'red', 'black', 'blue', 'orange', 'cyan']
    
    # Dynamically plot each cluster group based on the chosen K
    for cluster_id in range(k_value):
        cluster_data = df_processed[df_processed.cluster == cluster_id]
        ax_cluster.scatter(
            cluster_data['Age'], 
            cluster_data['Income($)'], 
            color=colors[cluster_id % len(colors)], 
            label=f'Cluster {cluster_id}',
            alpha=0.7
        )
        
    # Plotting Centroids
    if k_value > 1:
        centroids = km.cluster_centers_
        ax_cluster.scatter(
            centroids[:, 0], 
            centroids[:, 1], 
            color='magenta', 
            marker='*', 
            s=200, 
            label='Centroid'
        )
        
    ax_cluster.set_xlabel('Age (Scaled)' if apply_scaling else 'Age')
    ax_cluster.set_ylabel('Income (Scaled)' if apply_scaling else 'Income($)')
    ax_cluster.legend()
    ax_cluster.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_cluster)

# --- EXERCISE SECTION ---
st.markdown("---")
st.subheader("🎯 Try it Yourself (Exercise)")
st.write("""
1. Use the **Iris flower dataset** from the `sklearn.datasets` library (`load_iris`) and try to cluster flowers using petal width and length features.
2. Uncheck the **MinMaxScaler Preprocessing** checkbox on the left sidebar to see how badly a lack of feature scaling ruins the algorithm when features have wildly different ranges!
""")

