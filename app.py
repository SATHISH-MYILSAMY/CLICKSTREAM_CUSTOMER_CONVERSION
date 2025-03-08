import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


os.makedirs("mnt/data", exist_ok=True)

train_file_path = 'mnt/data/train_data.csv'
test_file_path = 'mnt/data/test_data.csv'

if os.path.exists(train_file_path) and os.path.exists(test_file_path):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
else:
    st.error("Train or Test dataset file not found. Please upload the correct files.")
    st.stop()

features = ['price', 'price_2', 'year', 'month', 'day']
target_classification = 'order'
target_regression = 'price_2'

X_train = train_df[features]
y_train_class = train_df[target_classification]
y_train_reg = train_df[target_regression]

X_test = test_df[features]
y_test_class = test_df[target_classification]
y_test_reg = test_df[target_regression]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_and_save_models():
    classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
    classification_model.fit(X_train, y_train_class)
    with open('mnt/data/classification_model.pkl', 'wb') as f:
        pickle.dump(classification_model, f)
    
    regression_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    regression_model.fit(X_train, y_train_reg)
    with open('mnt/data/regression_model.pkl', 'wb') as f:
        pickle.dump(regression_model, f)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_train)
    with open('mnt/data/clustering_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

if not os.path.exists('mnt/data/classification_model.pkl') or not os.path.exists('mnt/data/regression_model.pkl') or not os.path.exists('mnt/data/clustering_model.pkl'):
    train_and_save_models()

try:
    with open('mnt/data/classification_model.pkl', 'rb') as f:
        classification_model = pickle.load(f)
except FileNotFoundError:
    st.error("Classification model file not found. Please upload the correct model file.")
    classification_model = None

try:
    with open('mnt/data/regression_model.pkl', 'rb') as f:
        regression_model = pickle.load(f)
except FileNotFoundError:
    st.error("Regression model file not found. Please upload the correct model file.")
    regression_model = None

try:
    with open('mnt/data/clustering_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
except FileNotFoundError:
    st.error("Clustering model file not found. Please upload the correct model file.")
    kmeans = None

st.title("Customer Conversion Analysis")
st.sidebar.header("Upload Your Test CSV")

uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

if all(feature in data.columns for feature in features):
    X_test = data[features]
    X_test_scaled = scaler.transform(X_test)

else:
    st.error("Missing required feature columns in uploaded file.")
    st.stop()

st.sidebar.subheader("Manual Input for Predictions")
manual_input = {}
for feature in features:
    if feature == 'month':
        manual_input[feature] = st.sidebar.number_input(f"Enter {feature} (1-12)", min_value=1, max_value=12, value=1)
    elif feature == 'day':
        manual_input[feature] = st.sidebar.number_input(f"Enter {feature} (1-31)", min_value=1, max_value=31, value=1)
    elif feature == 'year':
        manual_input[feature] = st.sidebar.number_input(f"Enter {feature}", min_value=2000, max_value=2100, value=2022)
    else:
        manual_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)

if st.sidebar.button("Predict for Manual Input"):
    manual_df = pd.DataFrame([manual_input])
    manual_df_scaled = scaler.transform(manual_df)

    if classification_model and regression_model and kmeans:
        class_pred = classification_model.predict(manual_df)[0]
        reg_pred = regression_model.predict(manual_df)[0]
        cluster_pred = kmeans.predict(manual_df)[0]
        st.write(f"Predicted Conversion: {class_pred}")
        st.write(f"Predicted Revenue: {reg_pred}")
        st.write(f"Cluster Assignment: {cluster_pred}")
    else:
        st.error("One or more models are not loaded correctly. Cannot make predictions.")


st.sidebar.subheader("Make Predictions from Uploaded Data")
if st.sidebar.button("Predict Conversion & Revenue"):
    if classification_model and regression_model and kmeans:
        classification_prediction = classification_model.predict(X_test)
        regression_prediction = regression_model.predict(X_test)
        
        st.write("Checking test data before clustering:")
        st.write(X_test.head())
        
        if X_test.isnull().sum().sum() > 0:
            st.error("X_test contains missing values. Clustering cannot proceed.")
        else:
            try:
                cluster_labels = kmeans.predict(X_test)
                data['Cluster'] = cluster_labels
            except Exception as e:
                st.error(f"Clustering failed: {e}")
        
        data['Predicted_Conversion'] = classification_prediction
        data['Predicted_Revenue'] = regression_prediction
        
        st.write("Predictions:")
        if 'session_id' in data.columns:
            st.write(data[['session_id', 'Predicted_Conversion', 'Predicted_Revenue', 'Cluster']].head())
        else:
            st.write(data[['Predicted_Conversion', 'Predicted_Revenue', 'Cluster']].head())
    else:
        st.error("One or more models are not loaded correctly. Cannot make predictions.")
if kmeans:
    data['Cluster'] = kmeans.predict(X_test_scaled)

st.subheader("Data Visualizations")
if 'price' in data.columns:
    fig, ax = plt.subplots()
    sns.histplot(data['price'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

if 'Cluster' in data.columns:
    fig, ax = plt.subplots()
    sns.countplot(x='Cluster', data=data, ax=ax)
    st.pyplot(fig)

st.write("Clustering Results:")
if 'Cluster' in data.columns:
    if 'session_id' in data.columns:
        st.write(data[['session_id', 'Cluster']].head())
    else:
        st.write(data[['Cluster']].head())
else:
    st.write("Cluster predictions are not available.")