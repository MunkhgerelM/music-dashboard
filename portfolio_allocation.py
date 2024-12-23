import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime

# Streamlit App Configuration
st.set_page_config(page_title="Stock Prices Analysis", layout="wide")

# App Title
st.title("Stock Prices Prediction and Analysis")

# Sidebar Configuration
st.sidebar.header("Options")
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
selected_stocks = st.sidebar.multiselect("Select Stocks to Analyze", tech_list, default=tech_list)

start_date = st.sidebar.date_input(
    "Start Date", value=datetime(datetime.now().year - 1, datetime.now().month, datetime.now().day)
)
end_date = st.sidebar.date_input("End Date", value=datetime.now())

# Fetching Data
def fetch_data(tickers, start, end):
    """Fetch stock data for given tickers and date range."""
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)
        data['company_name'] = ticker
        all_data[ticker] = data
    return all_data

st.sidebar.write("Fetching stock data...")
try:
    stock_data = fetch_data(selected_stocks, start_date, end_date)
    st.sidebar.success("Data fetched successfully!")
except Exception as e:
    st.sidebar.error(f"Error fetching data: {e}")

# Data Display
st.subheader("Stock Data")
if stock_data:
    for ticker, data in stock_data.items():
        st.write(f"**{ticker}**: {data[['Close']].head(10)}")

# Plotting Closing Prices
st.subheader("Closing Prices")
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)
for i, (ticker, data) in enumerate(stock_data.items(), 1):
    plt.subplot(2, 2, i)
    data['Close'].plot(color='red')
    plt.ylabel('Close')
    plt.title(f"Closing Price of {ticker}")
plt.tight_layout()
st.pyplot(plt)

# Plotting Sales Volume
st.subheader("Sales Volume")
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)
for i, (ticker, data) in enumerate(stock_data.items(), 1):
    plt.subplot(2, 2, i)
    data['Volume'].plot(color='red')
    plt.ylabel('Volume')
    plt.title(f"Sales Volume for {ticker}")
plt.tight_layout()
st.pyplot(plt)

# Moving Averages
st.subheader("Moving Averages")
ma_day = [10, 20, 50]
for ma in ma_day:
    for data in stock_data.values():
        column_name = f"MA for {ma} days"
        data[column_name] = data['Close'].rolling(ma).mean()

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)
for ax, (ticker, data) in zip(axes.flatten(), stock_data.items()):
    data[['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=ax)
    ax.set_title(ticker)
fig.tight_layout()
st.pyplot(fig)

# Daily Returns
st.subheader("Daily Returns")
for data in stock_data.values():
    data['Daily Return'] = data['Close'].pct_change()

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)
for ax, (ticker, data) in zip(axes.flatten(), stock_data.items()):
    data['Daily Return'].plot(ax=ax, legend=True, linestyle='--', marker='o')
    ax.set_title(ticker)
fig.tight_layout()
st.pyplot(fig)

# Daily Returns Histogram
st.subheader("Daily Returns Histogram")
plt.figure(figsize=(12, 9))
for i, (ticker, data) in enumerate(stock_data.items(), 1):
    plt.subplot(2, 2, i)
    data['Daily Return'].hist(bins=50, color='red')
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(ticker)
plt.tight_layout()
st.pyplot(plt)


# Correlation Heatmaps
st.subheader("Correlation Heatmaps")

# Prepare valid closing prices
valid_closing_prices = {}
for ticker, data in stock_data.items():
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns and not data['Close'].empty:
        valid_closing_prices[ticker] = data['Close']

# Ensure all valid closing prices have the same index
if valid_closing_prices:
    closing_prices = pd.concat(valid_closing_prices.values(), axis=1, keys=valid_closing_prices.keys())
else:
    closing_prices = pd.DataFrame()  # Empty DataFrame if no valid data

if closing_prices.empty:
    st.write("No valid data available to compute correlations.")
else:
    # Compute correlations and plot heatmaps
    returns = closing_prices.pct_change()

    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    sns.heatmap(returns.corr(), annot=True, cmap='viridis')
    plt.title('Correlation of Stock Returns')

    plt.subplot(2, 2, 2)
    sns.heatmap(closing_prices.corr(), annot=True, cmap='viridis')
    plt.title('Correlation of Stock Closing Prices')

    st.pyplot(plt)
