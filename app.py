import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Title of the app
st.title("Machine Learning-Based Portfolio Optimization")

# Explanation section
st.markdown("""
### Overview:
This app predicts stock returns using Ridge Regression and optimizes portfolio allocation using mean-variance optimization.
You'll be able to select the assets, date range, and the app will guide you through return prediction and portfolio optimization.
""")

# User Input: Stock Tickers and Date Range
st.sidebar.header('User Input Parameters')
tickers = st.sidebar.multiselect('Select assets for your portfolio', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX'], default=['AAPL', 'GOOGL', 'MSFT'])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))

# Function to get stock data
def get_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

# Load and display stock data
if len(tickers) > 0:
    stock_prices, returns = get_stock_data(tickers, start_date, end_date)
    st.write(f"Displaying data for selected assets: {tickers}")
    st.line_chart(stock_prices)
else:
    st.write("Please select at least one stock.")

# Explanation of Return Prediction
st.markdown("""
### Step 1: Return Prediction
Using Ridge Regression, we predict the next period's returns based on historical data. Ridge helps prevent overfitting by applying regularization.
""")

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(returns.iloc[:-1], returns.iloc[1:], test_size=0.2, shuffle=False)

# Ridge regression for return prediction
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predicted_returns = model.predict(X_test)

# Show the prediction results
st.write("Predicted returns (on the test set):")
predicted_df = pd.DataFrame(predicted_returns, index=y_test.index, columns=tickers)
st.line_chart(predicted_df)

# Show actual vs predicted comparison
st.write("Actual vs Predicted Returns:")
for ticker in tickers:
    st.line_chart(pd.concat([y_test[ticker], predicted_df[ticker]], axis=1, keys=['Actual', 'Predicted']))

# Explanation of Portfolio Optimization
st.markdown("""
### Step 2: Portfolio Optimization
We use mean-variance optimization to find the best portfolio allocation that maximizes expected return while minimizing risk (measured by variance).
""")

# Mean returns and covariance matrix of the predicted returns
mean_returns = np.mean(predicted_returns, axis=0)
cov_matrix = np.cov(predicted_returns.T)

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

# Objective function: minimize the negative Sharpe ratio
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0):
    returns, risk = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe_ratio = (returns - risk_free_rate) / risk
    return -sharpe_ratio

# Constraints: weights sum to 1, no short selling
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Initial guess for weights
init_guess = len(tickers) * [1./len(tickers)]

# Optimize portfolio weights
opt_result = minimize(negative_sharpe, init_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x

# Display optimal weights
st.write("Optimal Portfolio Weights:")
st.write(dict(zip(tickers, optimal_weights)))

# Calculate portfolio performance
opt_returns, opt_risk = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
sharpe_ratio = (opt_returns) / opt_risk

st.write(f"Expected Return: {opt_returns*100:.2f}%")
st.write(f"Risk (Std Dev): {opt_risk*100:.2f}%")
st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix):
    frontier_returns = []
    frontier_risks = []
    for r in np.linspace(min(mean_returns), max(mean_returns), 100):
        constraints = [{'type': 'eq', 'fun': lambda w: portfolio_performance(w, mean_returns, cov_matrix)[0] - r},
                       {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1], init_guess, bounds=bounds, constraints=constraints)
        frontier_returns.append(r)
        frontier_risks.append(result.fun)
    return frontier_returns, frontier_risks

frontier_returns, frontier_risks = efficient_frontier(mean_returns, cov_matrix)

# Visualization of the efficient frontier
st.markdown("""
### Step 3: Efficient Frontier
The efficient frontier represents the optimal portfolios that provide the maximum return for a given level of risk.
""")
fig, ax = plt.subplots()
ax.plot(frontier_risks, frontier_returns, 'b--', label="Efficient Frontier")
ax.scatter(opt_risk, opt_returns, color='r', label="Optimal Portfolio", marker='x')
ax.set_xlabel('Risk (Standard Deviation)')
ax.set_ylabel('Expected Return')
ax.legend()
st.pyplot(fig)
