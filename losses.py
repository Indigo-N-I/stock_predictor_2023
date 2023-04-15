import tensorflow as tf

def expo_percent_error(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    pct_diff = diff / tf.clip_by_value(tf.abs(y_true), 1e-8, tf.float32.max) * 100.0

    # penalize more for large difference
    # penalize more for going above then going below
    loss_below = tf.exp(tf.clip_by_value(pct_diff / 100.0, -50, 50))
    loss_above = tf.exp(tf.clip_by_value(pct_diff / 50.0, -50, 50))
    return tf.where(y_pred < y_true, loss_below, loss_above)

def percentage_error(y_true, y_pred):
    """
    Computes the percentage error between y_true and y_pred.
    """
    return tf.reduce_mean(tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, tf.float32.max))) * 100.0
