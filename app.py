import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import streamlit as st
import kagglehub
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Download Dataset
# -------------------------------
path = kagglehub.dataset_download("chakramlops/rossmann-store-sales-dataset")
st.write("Dataset downloaded at:", path)

# Load CSV (adjust filename if different)
df = pd.read_csv(f"{path}/train.csv", parse_dates=['Date'])

# -------------------------------
# 2️⃣ Feature Engineering
# -------------------------------
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['IsWeekend'] = df['Weekday'].isin([5,6]).astype(int)

# Lag features
df = df.sort_values(['Store','Date'])
df['lag_1'] = df.groupby('Store')['Sales'].shift(1)
df['lag_7'] = df.groupby('Store')['Sales'].shift(7)
df['lag_14'] = df.groupby('Store')['Sales'].shift(14)

# Rolling features
df['rolling_mean_7'] = df.groupby('Store')['Sales'].shift(1).rolling(7).mean()
df['rolling_mean_30'] = df.groupby('Store')['Sales'].shift(1).rolling(30).mean()

# PromoInterval → IsPromoMonth
df['MonthName'] = df['Date'].dt.strftime('%b')
def is_promo_month(row):
    if pd.isna(row['PromoInterval']):
        return 0
    return 1 if row['MonthName'] in row['PromoInterval'] else 0
df['IsPromoMonth'] = df.apply(is_promo_month, axis=1)
df = df.drop(columns=['PromoInterval','MonthName'])

# One-hot encode categorical variables
categorical_cols = ['StoreType','Assortment','StateHoliday']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop NaN from lag features
df = df.dropna()

# -------------------------------
# 3️⃣ Train/Test Split
# -------------------------------
split_date = '2015-06-01'
train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

y_train = train_df['Sales']
y_test = test_df['Sales']
drop_cols = ['Sales','Date']
X_train = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=drop_cols)

# -------------------------------
# 4️⃣ Train XGBoost
# -------------------------------
st.write("Training XGBoost...")
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
st.write("Model trained!")

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# -------------------------------
# 5️⃣ Visualizations
# -------------------------------
st.write("Actual vs Predicted (first 200 days)")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y_test.values[:200], label='Actual')
ax.plot(y_pred[:200], label='Predicted')
ax.legend()
st.pyplot(fig)

# -------------------------------
# 6️⃣ Predict Sales for New Input
# -------------------------------
st.write("Predict Sales for a Store")
store = st.number_input("Store ID", min_value=int(df['Store'].min()), max_value=int(df['Store'].max()), value=1)
date_input = st.date_input("Date")
promo = st.checkbox("Promo?")

# Feature prep for new row
new_row = pd.DataFrame({
    'Store': [store],
    'Promo': [int(promo)],
    'Year': [date_input.year],
    'Month': [date_input.month],
    'Day': [date_input.day],
    'Weekday': [date_input.weekday()],
    'IsWeekend': [1 if date_input.weekday() in [5,6] else 0],
    # Lag features set as previous day sales (simplest for demo)
    'lag_1': [X_test.loc[X_test['Store']==store, 'lag_1'].iloc[-1]],
    'lag_7': [X_test.loc[X_test['Store']==store, 'lag_7'].iloc[-1]],
    'lag_14': [X_test.loc[X_test['Store']==store, 'lag_14'].iloc[-1]],
    'rolling_mean_7': [X_test.loc[X_test['Store']==store, 'rolling_mean_7'].iloc[-1]],
    'rolling_mean_30': [X_test.loc[X_test['Store']==store, 'rolling_mean_30'].iloc[-1]],
    'IsPromoMonth': [1 if date_input.strftime('%b') in df.loc[df['Store']==store,'PromoInterval'].dropna().values else 0]
})

# Handle categorical columns
for col in X_train.columns:
    if col not in new_row.columns:
        new_row[col] = 0

y_new_pred = xgb.predict(new_row[X_train.columns])
st.write(f"Predicted Sales: {y_new_pred[0]:.2f}")
