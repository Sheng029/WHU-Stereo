import tensorflow as tf
import tensorflow.keras as keras


class Estimation(keras.Model):
    def __init__(self, min_disp=-112.0, max_disp=16.0):
        super(Estimation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.conv = keras.layers.Conv3D(filters=1, kernel_size=3,
                                        strides=1, padding='same',
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(1.0e-5))

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)     # [N, D, H, W, 1]
        x = tf.squeeze(x, -1)     # [N, D, H, W]
        x = tf.transpose(x, (0, 2, 3, 1))     # [N, H, W, D]
        assert x.shape[-1] == self.max_disp - self.min_disp
        candidates = tf.linspace(1.0 * self.min_disp, 1.0 * self.max_disp - 1.0, self.max_disp - self.min_disp)
        probabilities = tf.math.softmax(-1.0 * x, -1)
        disparities = tf.reduce_sum(candidates * probabilities, -1, True)
        return disparities
