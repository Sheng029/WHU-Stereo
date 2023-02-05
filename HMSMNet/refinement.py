import tensorflow as tf
import tensorflow.keras as keras
from feature import conv2d, L2


def conv_bn_act(filters, kernel_size, strides, padding, dilation_rate):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               dilation_rate=dilation_rate, use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    act = keras.layers.LeakyReLU()

    return keras.Sequential([conv, bn, act])


class Refinement(keras.Model):
    def __init__(self, filters):
        super(Refinement, self).__init__()
        self.conv1 = conv_bn_act(filters, 3, 1, 'same', 1)
        self.conv2 = conv_bn_act(filters, 3, 1, 'same', 1)
        self.conv3 = conv_bn_act(filters, 3, 1, 'same', 2)
        self.conv4 = conv_bn_act(filters, 3, 1, 'same', 3)
        self.conv5 = conv_bn_act(filters, 3, 1, 'same', 1)
        self.conv6 = conv2d(1, 3, 1, 'same', 1)

    def call(self, inputs, training=None, mask=None):
        # inputs: [disparity, rgb, gx, gy]
        assert len(inputs) == 4

        scale_factor = inputs[1].shape[1] / inputs[0].shape[1]
        disp = tf.image.resize(inputs[0], [inputs[1].shape[1], inputs[1].shape[2]])
        disp = disp * scale_factor

        concat = tf.concat([disp, inputs[1], inputs[2], inputs[3]], -1)
        delta = self.conv1(concat)
        delta = self.conv2(delta)
        delta = self.conv3(delta)
        delta = self.conv4(delta)
        delta = self.conv5(delta)
        delta = self.conv6(delta)
        disp_final = disp + delta

        return disp_final
