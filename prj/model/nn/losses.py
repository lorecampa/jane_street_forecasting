import tensorflow as tf

class WeightedZeroMeanR2Loss(tf.keras.losses.Loss):
    def __init__(self, name="weighted_zero_mean_r2_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true)            
        return tf.reduce_sum(sample_weight * tf.square(y_true - y_pred)) / tf.maximum(tf.reduce_sum(sample_weight * tf.square(y_true)), 1e-7)
