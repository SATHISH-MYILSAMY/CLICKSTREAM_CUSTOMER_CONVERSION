import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans

# Streamlit UI Setup
st.title("Customer Conversion Analysis")
st.sidebar.header("Upload Your CSV")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
else:
    st.write("Please upload a CSV file to proceed.")
    st.stop()

# Define features and target
features = ['price', 'price_2', 'year', 'month', 'day']
target_classification = 'order'
target_regression = 'price_2'

X = data[features]
y_classification = data[target_classification]
y_regression = data[target_regression]

# Train-test split
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Define classification model
classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
classification_model.fit(X_train, y_train_class)
y_pred_classification = classification_model.predict(X_test)

# Define regression model
regression_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
regression_model.fit(X_train_reg, y_train_reg)
y_pred_regression = regression_model.predict(X_test_reg)

# Define clustering model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.predict(X)

data['Cluster'] = kmeans_labels

# Real-time Prediction Input
st.sidebar.subheader("Make a Prediction")
user_input = {col: st.sidebar.number_input(col, value=0.0) for col in features}
input_df = pd.DataFrame([user_input])

if st.sidebar.button("Predict Conversion & Revenue"):
    classification_prediction = classification_model.predict(input_df)
    regression_prediction = regression_model.predict(input_df)
    st.write("Predicted Conversion:", classification_prediction[0])
    st.write("Predicted Revenue:", regression_prediction[0])

# Visualization
st.subheader("Data Visualizations")
fig, ax = plt.subplots()
sns.histplot(data['price'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.countplot(x='Cluster', data=data, ax=ax)
st.pyplot(fig)

st.write("Clustering Results:")
st.write(data[['session_id', 'Cluster']].head())
