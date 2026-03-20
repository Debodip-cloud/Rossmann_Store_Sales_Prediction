import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import streamlit as st
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1️⃣ Download Dataset
# -------------------------------
path = kagglehub.dataset_download("chakramlops/rossmann-store-sales-dataset")
st.write("Dataset downloaded at:", path)

# Load CSV files
train_df = pd.read_csv(f"{path}/train.csv", parse_dates=['Date'])
store_df = pd.read_csv(f"{path}/store.csv")

# Merge train + store info
df = train_df.merge(store_df, on='Store', how='left')

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
    return 1 if row['MonthName'] in str(row['PromoInterval']).split(',') else 0
df['IsPromoMonth'] = df.apply(is_promo_month, axis=1)
df = df.drop(columns=['PromoInterval','MonthName'])

# One-hot encode categorical variables
categorical_cols = ['StoreType','Assortment','StateHoliday']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop NaN from lag/rolling features
df = df.dropna()

# -------------------------------
# 3️⃣ Train/Test Split
# -------------------------------
split_date = '2015-06-01'
train_data = df[df['Date'] < split_date]
test_data = df[df['Date'] >= split_date]

y_train = train_data['Sales']
y_test = test_data['Sales']
drop_cols = ['Sales','Date']
X_train = train_data.drop(columns=drop_cols)
X_test = test_data.drop(columns=drop_cols)

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

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# -------------------------------
# 5️⃣ Feature Importance
# -------------------------------
st.subheader("Top 10 Important Features")
importances = pd.Series(xgb.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False).head(10)
st.bar_chart(importances)

# -------------------------------
# 6️⃣ Visualizations
# -------------------------------
st.subheader("Sales Visualizations")

# 6a. Actual vs Predicted (first 200 days)
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(y_test.values[:200], label='Actual')
ax.plot(y_pred[:200], label='Predicted')
ax.set_title("Actual vs Predicted Sales (first 200 days)")
ax.legend()
st.pyplot(fig)

# 6b. Monthly Average Sales
monthly_sales = df.groupby(['Year','Month'])['Sales'].mean().reset_index()
monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + "-" + monthly_sales['Month'].astype(str)
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=monthly_sales, x='YearMonth', y='Sales', marker='o', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title("Monthly Average Sales")
st.pyplot(fig)

# 6c. Weekly Average Sales
weekly_sales = df.groupby('Weekday')['Sales'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=weekly_sales, x='Weekday', y='Sales', palette='Blues_d', ax=ax)
ax.set_title("Average Sales by Weekday (0=Mon, 6=Sun)")
st.pyplot(fig)

# 6d. Promo Impact
promo_sales = df.groupby('IsPromoMonth')['Sales'].mean().reset_index()
fig, ax = plt.subplots(figsize=(6,5))
sns.barplot(data=promo_sales, x='IsPromoMonth', y='Sales', palette='Oranges', ax=ax)
ax.set_title("Sales with Promo vs No Promo")
st.pyplot(fig)

# 6e. Top Stores by Total Sales
top_stores = df.groupby('Store')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=top_stores, x='Store', y='Sales', palette='Greens', ax=ax)
ax.set_title("Top 10 Stores by Total Sales")
st.pyplot(fig)

# -------------------------------
# 7️⃣ Predict Sales for New Input
# -------------------------------
st.subheader("Predict Sales for a Store")
store = st.number_input(
    "Store ID",
    min_value=int(df['Store'].min()),
    max_value=int(df['Store'].max()),
    value=int(df['Store'].min())
)
date_input = st.date_input("Date")
promo = st.checkbox("Promo?")

# Get PromoInterval for the store from store_df
promo_interval = store_df.loc[store_df['Store']==store, 'PromoInterval'].values
if len(promo_interval) == 0 or pd.isna(promo_interval[0]):
    is_promo_month_val = 0
else:
    months_list = str(promo_interval[0]).split(',')
    is_promo_month_val = 1 if date_input.strftime('%b') in months_list else 0

# Prepare lag/rolling features
last_store_data = df[df['Store']==store].sort_values('Date')
lag_1 = last_store_data['Sales'].iloc[-1]
lag_7 = last_store_data['Sales'].iloc[-7] if len(last_store_data) >= 7 else lag_1
lag_14 = last_store_data['Sales'].iloc[-14] if len(last_store_data) >= 14 else lag_1
rolling_mean_7 = last_store_data['Sales'].iloc[-7:].mean() if len(last_store_data) >=7 else lag_1
rolling_mean_30 = last_store_data['Sales'].iloc[-30:].mean() if len(last_store_data) >=30 else lag_1

# Build new row
new_row = pd.DataFrame({
    'Store': [store],
    'Promo': [int(promo)],
    'Year': [date_input.year],
    'Month': [date_input.month],
    'Day': [date_input.day],
    'Weekday': [date_input.weekday()],
    'IsWeekend': [1 if date_input.weekday() in [5,6] else 0],
    'lag_1': [lag_1],
    'lag_7': [lag_7],
    'lag_14': [lag_14],
    'rolling_mean_7': [rolling_mean_7],
    'rolling_mean_30': [rolling_mean_30],
    'IsPromoMonth': [is_promo_month_val]
})

# Add missing one-hot columns
for col in X_train.columns:
    if col not in new_row.columns:
        new_row[col] = 0

y_new_pred = xgb.predict(new_row[X_train.columns])
st.write(f"Predicted Sales: {y_new_pred[0]:.2f}")
