import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from datagenerator import DataGenerator
from models import tcn_model
from tensorflow.keras.models import load_model
from losses import percentage_error, expo_percent_error
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler

class AvgPercentageError(keras.metrics.Metric):
    def __init__(self, name='avg_perc_error', **kwargs):
        super(AvgPercentageError, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        diff = y_true - y_pred
        pct_diff = tf.divide(diff, tf.clip_by_value(tf.abs(y_true), 1e-8, tf.float32.max)) * 100.0
        self.total.assign_add(tf.reduce_sum(pct_diff))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

def cyclical_lr(epoch, lr):
    if epoch % 10 == 0:
        lr *= 1.1
    else:
        lr *= .9
    return lr

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)
# if tf.test.is_gpu_available():
#     print('GPU is available')
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     tf.config.experimental.set_virtual_device_configuration(
#         physical_devices[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)])
# else:
#     print('GPU is not available')
load = True

train_data = pd.read_csv('sp500_train.csv', index_col=0)

# Create data generator instance
train_generator = DataGenerator(train_data, batch_size=32)


custom_objects = {'expo_percent_error': expo_percent_error, "AvgPercentageError": AvgPercentageError, "cyclical_lr": cyclical_lr}
if not load:
    # Create and compile TCN model
    model = tcn_model()
else:
    model = load_model('best_model_exp-08-1.2741.h5', custom_objects=custom_objects)

# Define checkpoint filepath
filepath="best_model_exp-{epoch:02d}-{loss:.4f}.h5"

lr_scheduler = LearningRateScheduler(cyclical_lr)

# Define model checkpoint callback to save the best model
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)

opt = Adam(lr=0.001)
model.compile(loss=expo_percent_error, optimizer=opt, metrics = [AvgPercentageError()])

# Train TCN model
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
model.fit(train_generator, epochs=100, callbacks=[es, checkpoint,lr_scheduler])
#219,544,207,360.0000
#6269953246232576
#10437021784866816
