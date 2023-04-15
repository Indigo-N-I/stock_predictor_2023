import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def tcn_model():
    # Input layer
    input_layer = Input(shape=(30, 93))

    # TCN pathway 1
    tcn1 = Conv1D(32, kernel_size=3, activation='relu', padding='causal')(input_layer)
    tcn1 = Dropout(0.2)(tcn1)
    tcn1 = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(tcn1)
    tcn1 = Dropout(0.2)(tcn1)
    tcn1 = Conv1D(128, kernel_size=3, activation='relu', padding='causal')(tcn1)
    tcn1 = Dropout(0.2)(tcn1)
    tcn1 = Conv1D(256, kernel_size=3, activation='relu', padding='causal')(tcn1)
    tcn1 = Dropout(0.2)(tcn1)
    tcn1 = Flatten()(tcn1)

    # TCN pathway 2
    tcn2 = Conv1D(32, kernel_size=5, activation='tanh', padding='causal')(input_layer)
    tcn2 = Dropout(0.2)(tcn2)
    tcn2 = Conv1D(64, kernel_size=5, activation='tanh', padding='causal')(tcn2)
    tcn2 = Dropout(0.2)(tcn2)
    tcn2 = Conv1D(128, kernel_size=5, activation='tanh', padding='causal')(tcn2)
    tcn2 = Dropout(0.2)(tcn2)
    tcn2 = Conv1D(256, kernel_size=5, activation='tanh', padding='causal')(tcn2)
    tcn2 = Dropout(0.2)(tcn2)
    tcn2 = Flatten()(tcn2)

    # Fully connected pathway
    fc = Flatten()(input_layer)
    fc = Dense(256, activation='elu')(fc)
    fc = Dropout(0.2)(fc)
    fc = Dense(128, activation='elu')(fc)
    fc = Dropout(0.2)(fc)
    fc = Dense(64, activation='elu')(fc)
    fc = Dropout(0.2)(fc)

    # Merge all pathways
    merged = concatenate([tcn1, tcn2, fc])
    merged = Dense(500, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.2)(merged)

    # Output layer
    output_layer = Dense(1, activation='linear')(merged)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
