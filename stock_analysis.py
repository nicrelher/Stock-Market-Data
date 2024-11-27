import yfinance as yf # imports data from yahoo finance
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# DOWNLOADS 2019 - 2023 Apple stock data
stock_data = yf.download("AAPL", start = "2019-01-01", end = "2023-12-31")
print(stock_data.head()) # '.head() is a method used with Pandas DataFrames and Series to view the first fiew rows of data (default is 5 rows)'

# PLOT STOCK PRICE
stock_data['Close'].plot(figsize = (12, 6)) # close = closing price
plt.title('Apple Stock Price (2019 - 2023)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()

# CALCULATES DAILY RETURNS
stock_data['Daily_Return'] = stock_data['Close'].pct_change()
# PLOTS DAILY RETURNS
stock_data['Daily_Return'].plot(figsize = (12, 6))
plt.title('Apple Daily Returns (2019 - 2023)')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.show()

# MOVING AVERAGES (50-day moving average and 200-day moving average)
stock_data['50_MA'] = stock_data['Close'].rolling(window = 50).mean()
stock_data['200_MA'] = stock_data['Close'].rolling(window = 200).mean()
# PLOT MOVING AVERAGE
stock_data[['Close', '50_MA', '200_MA']].plot(figsize = (12, 6))
plt.title('Apple Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.show()

# USES LINEAR REGRESSION; PREPARES THE DATA FOR PREDICTION
stock_data = stock_data.dropna() # Drops rows with missing values
x = stock_data[['50_MA', '200_MA']] # features (50 and 200-day moving averages)
y = stock_data['Close'].shift(-1) # Target (next day's closing price)
# SPLITS DATA INTO TRAINING AND TESTING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

# (fit the model on the training data) USES LINEAR REGRESSION
from sklearn.linear_model import LinearRegression # sklearn is for machine learning, performs data preprocessing and evluating performance
model = LinearRegression()
model.fit(X_train, y_train)

# MAKES PREDICTIONS (use trained model to make predictions on the test data)
predictions = model.predict(X_test)
# EVALUATES MODEL; CALCULATES THE MEAN ABSOLUTE ERROR TO EVALUATE THE MODEL'S PERFORMANCE
from sklearn.metrics import mean_absolute_error
print(f"MAE: {mean_absolute_error(y_test, predictions)}")

# PORTFOLIO OPTIMIZATION (Markowitz portfolio theory)
# Downloads data for multiple stocks
tickers = ['AAPL', 'MSFT', 'GOOG']
stock_data_multiple = yf.download(tickers, start = '2019-01-01', end = '2023-12-31')
# CALCULATES RETURNS AND CORRELATION
returns = stock_data_multiple.pct_change()
correlation_matrix = returns.corr()
returns.plot(figsize = (12, 6))
plt.title('Stock Returns (2019-2023)')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.show()
