🛒 Rossmann Store Sales Prediction

This project predicts daily sales for Rossmann stores using machine learning and time series feature engineering. It also includes an interactive web app built with Streamlit.

📌 Project Overview

The goal of this project is to forecast store sales based on historical data, promotions, and seasonal patterns.

This is a real-world business problem where accurate predictions can help:

Improve inventory planning

Optimize staffing

Increase revenue through better decisions

📊 Dataset

Rossmann Store Sales dataset (Kaggle)

Includes:

Daily sales data

Store information

Promotions and holidays

⚙️ Features & Techniques
🔹 Feature Engineering

Date features (Year, Month, Day, Weekday)

Weekend indicator

Lag features (1, 7, 14 days)

Rolling averages (7, 30 days)

Promo month extraction from PromoInterval

One-hot encoding for categorical variables

🔹 Model

XGBoost Regressor

🔹 Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

📈 Results

Linear Regression RMSE: ~976

XGBoost RMSE: ~466

👉 XGBoost significantly improved performance by capturing non-linear patterns.

📊 Visualizations

The project includes:

Actual vs Predicted Sales

Monthly Sales Trends

Weekly Sales Patterns

Promotion Impact on Sales

Top Performing Stores

🖥️ Streamlit App

The app allows you to:

Train the model

Visualize results

Predict sales for any store and date

Input:

Store ID

Date

Promotion status

Output:

Predicted Sales

🚀 How to Run Locally
git clone https://github.com/your-username/rossmann_store_sales_prediction.git
cd rossmann_store_sales_prediction
pip install -r requirements.txt
streamlit run app.py
⚠️ Deployment Note

For deployment (Streamlit Cloud):

Dataset is stored locally in the repo

Avoid external downloads (e.g., Kaggle API) for reliability

💡 Key Insights

Lag features are the most important → past sales strongly influence future sales

Promotions significantly increase sales

Rolling averages capture seasonality trends

Weekly patterns show clear variation (weekends vs weekdays)

🧠 What I Learned

Time series feature engineering

Handling real-world messy data

Model evaluation for forecasting

Building end-to-end ML apps with Streamlit

📬 Contact

If you’d like to discuss this project or opportunities in Data Science / FinTech, feel free to connect!
