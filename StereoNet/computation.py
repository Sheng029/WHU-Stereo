import tensorflow as tf
import tensorflow.keras as keras


class Computation(keras.Model):
    def __init__(self, min_disp, max_disp):
        super(Computation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def call(self, inputs, training=None, mask=None):
        assert inputs.shape[-1] == self.max_disp - self.min_disp
        candidates = tf.linspace(1.0 * self.min_disp, 1.0 * self.max_disp - 1.0, self.max_disp - self.min_disp)
        probabilities = tf.math.softmax(-1.0 * inputs, -1)
        disparities = tf.reduce_sum(candidates * probabilities, -1, True)
        return disparities
