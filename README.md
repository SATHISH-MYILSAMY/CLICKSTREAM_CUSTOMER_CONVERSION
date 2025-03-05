🔹 Data Preprocessing & Cleaning

1) Dataset Used: Train.csv for model training and Test.csv for validation.
2) Handling Missing Values: Used mean/median for numerical features and mode for categorical.
3) Feature Encoding: Applied One-Hot Encoding and Label Encoding for categorical variables.
4) Feature Scaling: Standardized numerical data using MinMaxScaler or StandardScaler.

🔹 Exploratory Data Analysis (EDA)

5) Data Visualization: Used bar charts, histograms, and pair plots for insights.
6) Session Analysis: Examined session duration, page views, and bounce rates.
7) Correlation Analysis: Identified relationships between variables via heatmaps.
8) Time-based Analysis: Extracted hourly, daily, and seasonal trends in user behavior.

🔹 Feature Engineering

9) Session Metrics: Created session length, click counts, and time spent per category.
10) Clickstream Patterns: Tracked user browsing paths to uncover behavioral trends.
11) Behavioral Metrics: Included bounce rates, exit rates, and revisit patterns.

🔹 Balancing Techniques for Classification

12) Imbalance Check: Analyzed class distribution (Converted vs. Not Converted).
13) Oversampling: Used SMOTE to generate synthetic samples for minority class.
14) Undersampling: Randomly removed majority class samples for balance.
15) Class Weight Adjustment: Adjusted weights during model training.

🔹 Model Building

16) Classification Models: Implemented Logistic Regression, Decision Trees, Random Forest, XGBoost, and Neural Networks.
17) Regression Models: Used Linear Regression, Ridge, Lasso, Gradient Boosting Regressor for revenue prediction.
18) Clustering Models: Performed K-Means, DBSCAN, and Hierarchical Clustering for segmentation.
19) Pipeline Automation: Used Scikit-learn Pipelines for data preprocessing, scaling, model training, and evaluation.

🔹 Model Evaluation & Deployment

20) Streamlit App: Developed an interactive web app for real-time predictions, revenue estimation, and clustering visualizations with CSV upload and dynamic charts.
