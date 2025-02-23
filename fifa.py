import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Load the dataset
fifa_data = pd.read_csv(r"C:\Users\nivet\OneDrive\Documents\projects_resume\fifa dataset.csv")

# Data Preprocessing
fifa_data['rank_date'] = pd.to_datetime(fifa_data['rank_date'])
fifa_data.fillna(method='ffill', inplace=True)
fifa_data['Year'] = fifa_data['rank_date'].dt.year

# EDA (Exploratory Data Analysis)
st.title("FIFA Dataset EDA and Ranking Prediction")

st.header("Summary Statistics")
st.write(fifa_data.describe())

st.header("Missing Values")
st.write(fifa_data.isnull().sum())

# Rank Frequency
st.header("Rank Frequency")
rank_frequency = fifa_data['rank'].value_counts()
st.write(rank_frequency)

# Plot: Distribution of FIFA Rankings
st.subheader("Distribution of FIFA Rankings")
fig, ax = plt.subplots()
sns.histplot(fifa_data['rank'], bins=30, color='skyblue', kde=True, ax=ax)
ax.set_title('Distribution of FIFA Rankings')
ax.set_xlabel('Rank')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Plot: Trend of FIFA Rankings Over the Years
st.subheader("Trend of FIFA Rankings Over the Years")
fig, ax = plt.subplots()
sns.lineplot(data=fifa_data, x='Year', y='rank', ci=None, ax=ax)
ax.set_title('Trend of FIFA Rankings Over the Years')
ax.set_xlabel('Year')
ax.set_ylabel('Rank')
st.pyplot(fig)

# Plot: FIFA Rankings by Confederation
st.subheader("FIFA Rankings by Confederation")
fig, ax = plt.subplots()
sns.boxplot(x='confederation', y='rank', data=fifa_data, ax=ax)
ax.set_title('FIFA Rankings by Confederation')
ax.set_xlabel('Confederation')
ax.set_ylabel('Rank')
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_cols = fifa_data.select_dtypes(include=[np.number]).columns.tolist()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(fifa_data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Select Features and Target
X = fifa_data[['total_points', 'previous_points', 'confederation', 'Year']]  # Initial features
y = fifa_data['rank']

# One-hot encode the confederation feature
X = pd.get_dummies(X, columns=['confederation'], drop_first=True)

# Feature Selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)

# Create DataFrame for feature scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
})

# Display feature scores
st.subheader("Feature Scores")
st.write(feature_scores.sort_values(by='Score', ascending=False))

# Keep only top features based on scores
top_features = feature_scores.nlargest(3, 'Score')['Feature'].values
X = X[top_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Model evaluation
def evaluate_model(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred)
    }

# Evaluate Random Forest model
st.subheader("Model Performance: Random Forest")
y_pred_rf = rf_model.predict(X_test_scaled)
rf_metrics = evaluate_model(y_test, y_pred_rf)
st.write(rf_metrics)

# Evaluate Gradient Boosting model
st.subheader("Model Performance: Gradient Boosting")
y_pred_gb = gb_model.predict(X_test_scaled)
gb_metrics = evaluate_model(y_test, y_pred_gb)
st.write(gb_metrics)

# Save all models and related data
joblib.dump(rf_model, 'random_forest_fifa_model.pkl')
joblib.dump(gb_model, 'gradient_boosting_fifa_model.pkl')
joblib.dump(scaler, 'fifa_standard_scaler.pkl')

# Streamlit interface for predictions
st.header("FIFA Ranking Prediction")

country_name = st.text_input("Enter the country name:")
total_points = st.number_input("Enter the total points:", min_value=0.0)
previous_points = st.number_input("Enter the previous points:", min_value=0.0)
confederation = st.selectbox("Select confederation:", fifa_data['confederation'].unique())

# Prediction function
def predict_fifa_ranking(total_points, previous_points, confederation):
    country_data = {
        'total_points': total_points,
        'previous_points': previous_points,
        'Year': fifa_data['Year'].max(),  # Assuming current year for prediction
        'confederation': confederation
    }

    country_df = pd.DataFrame([country_data])
    country_df = pd.get_dummies(country_df, columns=['confederation'], drop_first=True)

    for col in top_features:
        if col not in country_df.columns:
            country_df[col] = 0

    country_df = country_df[top_features]
    country_df_scaled = scaler.transform(country_df)

    predicted_rank_rf = rf_model.predict(country_df_scaled)[0]
    predicted_rank_gb = gb_model.predict(country_df_scaled)[0]

    return predicted_rank_rf, predicted_rank_gb

# Make prediction
if st.button("Predict FIFA Ranking"):
    predicted_rank_rf, predicted_rank_gb = predict_fifa_ranking(total_points, previous_points, confederation)

    st.write(f"Predicted FIFA Ranking for {country_name} (Random Forest): {predicted_rank_rf:.2f}")
    st.write(f"Predicted FIFA Ranking for {country_name} (Gradient Boosting): {predicted_rank_gb:.2f}")

    # Save the prediction output
    prediction_output = {
        'Country': country_name,
        'Random Forest Prediction': predicted_rank_rf,
        'Gradient Boosting Prediction': predicted_rank_gb
    }
    joblib.dump(prediction_output, 'fifa_prediction_output.pkl')
    st.write("Prediction output saved as 'fifa_prediction_output.pkl'.")