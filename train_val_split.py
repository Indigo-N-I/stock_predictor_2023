import pandas as pd
from sklearn.model_selection import train_test_split

# Read in data from CSV file
sp500_data = pd.read_csv('sp500_daily_movements_ta.csv', index_col=0)

# Split tickers into train/validation sets
train_tickers, val_tickers = train_test_split(sp500_data['Ticker'].unique(), test_size=0.2)

# Split data for each ticker into train/validation sets
train_data = sp500_data[sp500_data['Ticker'].isin(train_tickers)]
val_data = sp500_data[sp500_data['Ticker'].isin(val_tickers)]

# Save train/validation sets to separate CSV files
train_data.to_csv('sp500_train.csv')
val_data.to_csv('sp500_val.csv')
