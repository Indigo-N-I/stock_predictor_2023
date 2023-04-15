import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf


# Create data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.tickers = data['Ticker'].unique()
        self.indexes = {}
        for ticker in self.tickers:
            ticker_data = data[data['Ticker'] == ticker]
            ticker_data = ticker_data.drop(columns=['Ticker'])
            ticker_data = ticker_data.to_numpy(dtype='float32')
            self.indexes[ticker] = []
            for i in range(30, len(ticker_data)):
                self.indexes[ticker].append(i-30)
        self.lengths = [len(self.indexes[ticker]) for ticker in self.tickers]
        self.data = self.data.drop(columns = ['Ticker'])

    def __len__(self):
        return int(sum(np.ceil(length / self.batch_size) for length in self.lengths))

    def __getitem__(self, index):
        ticker_idx = 0
        # self.lengths[ticker_idx] is how long the ticker has
        while index >= np.ceil(self.lengths[ticker_idx] / self.batch_size):
            index -= np.ceil(self.lengths[ticker_idx] / self.batch_size)
            ticker_idx += 1
        ticker = self.tickers[ticker_idx]
        idx = int(index * self.batch_size)
        batch_indexes = self.indexes[ticker][idx:idx+self.batch_size]
        # print(batch_indexes)
        # print([self.data.loc[i:i+29] for i in batch_indexes])
        batch_x = np.array([self.data.loc[i:i+29].to_numpy(dtype=np.float32) for i in batch_indexes])
        batch_y = np.array([self.data.loc[i+29].to_numpy(dtype=np.float32)[3] for i in batch_indexes])
        # print(f"index:{index}, idx:{idx}, batch:{batch_indexes}, ticker_idx:{ticker_idx}, ticker:{ticker}, ticker length:{len(self.indexes[ticker])}")
        # print(batch_x.shape, batch_y.shape)
        return batch_x, batch_y
