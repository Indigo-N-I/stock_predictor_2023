import yfinance as yf
import pandas as pd
import ta
import numpy as np

# class DMIndicator(TrendIndicatorMixin, BaseIndicator):
#     high = "high"
#     low = "low"
#
#     def __init__(self, high=None, low=None, window=None, fillna=False, **kwargs):
#         self.high = high
#         self.low = low
#         self.window = window
#         self.fillna = fillna
#
#         self._means = None
#         self._dms = None
#
#         super().__init__(**kwargs)
#
#     def _check_fillna(self, series, value):
#         if self.fillna:
#             series = series.replace([np.inf, -np.inf], np.nan).fillna(value)
#         return series
#
#     def _calculate_means(self, high, low, window):
#         dmh = high - high.shift(1)
#         dml = low.shift(1) - low
#
#         self._dms = pd.concat([dmh, dml], axis=1)
#         self._dms[self._dms < 0] = 0
#
#         self._means = (
#             self._dms.rolling(window=window, min_periods=1).mean()
#             / pd.Series([window] * len(high), index=high.index).rolling(window=window, min_periods=1).mean()
#         )
#
#     def _calculate(self):
#         self._calculate_means(
#             high=self._check_fillna(self._df[self.high], method="ffill").astype(float),
#             low=self._check_fillna(self._df[self.low], method="ffill").astype(float),
#             window=self.window,
#         )
#
#         self._df[f"pdi_{self.window}"] = 100 * self._means.iloc[:, 0]
#         self._df[f"mdi_{self.window}"] = 100 * self._means.iloc[:, 1]
#         self._df[f"dx_{self.window}"] = 100 * abs(self._means.iloc[:, 0] - self._means.iloc[:, 1]) / (
#             self._means.iloc[:, 0] + self._means.iloc[:, 1]
#         )
#         self._df[f"adx_{self.window}"] = self._df[f"dx_{self.window}"].rolling(window=self.window).mean()
#
#         # This line has been modified to handle divide by zero warnings
#         self._df[f"adxr_{self.window}"] = (
#             self._df[f"adx_{self.window}"].shift(self.window // 2) + self._df[f"adx_{self.window}"]
#         ) / 2
#
#         self._df.drop(columns=["_dms", "_means"], inplace=True)
#         self._df.fillna(0, inplace=True)
pd.options.display.max_columns = None
if __name__ == "__main__":
    # Get list of all S&P 500 tickers
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

    # Download daily stock movements for each ticker and calculate TA indicators
    sp500_data = pd.DataFrame()
    for ticker in sp500_tickers:
        try:
            data = yf.download(ticker, start='2013-01-01', end='2023-04-11')
            data['Ticker'] = ticker
            print(f'Retrieved {ticker}')

            with pd.option_context('mode.chained_assignment', None):
                data.loc[:, 'log_return'] = np.log(data['Close'] / data['Close'].shift())
                data.fillna(data.mean(numeric_only=True), inplace=True)

            data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True, colprefix='ta_')
            print(f"TA for {ticker} finished")
            sp500_data = pd.concat([sp500_data, data], ignore_index=True)

        except Exception as e:
            print(f"Error while retrieving data for {ticker}: {str(e)}")

        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Terminating loop...")
            break

    # Save data to CSV file
    sp500_data.to_csv('sp500_daily_movements_ta.csv')
