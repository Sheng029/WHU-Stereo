import tensorflow as tf
import tensorflow.keras as keras
from feature import Conv3x3, ResidualBlock


L2 = 1.0e-4
alpha = 0.2


class Conv3x3BnLeakyRelu(keras.Model):
    """
    Conv3x3 followed by bn and leaky relu.
    """
    def __init__(self, filters):
        super(Conv3x3BnLeakyRelu, self).__init__()
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                                        activation=None, use_bias=False, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))
        self.bn = keras.layers.BatchNormalization()
        self.leaky_relu = keras.layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.leaky_relu(outputs)

        return outputs


class Refinement(keras.Model):
    """
    Disparity refinement used in StereoNet.
    """
    def __init__(self, filters):
        super(Refinement, self).__init__()
        # Conv, bn, activation for the concatenated color and disparity
        self.conv3x3 = Conv3x3BnLeakyRelu(filters=filters)
        # 6 dilated residual blocks
        self.res_1 = ResidualBlock(filters=filters, dilation_rate=1)
        self.res_2 = ResidualBlock(filters=filters, dilation_rate=2)
        self.res_3 = ResidualBlock(filters=filters, dilation_rate=4)
        self.res_4 = ResidualBlock(filters=filters, dilation_rate=8)
        self.res_5 = ResidualBlock(filters=filters, dilation_rate=1)
        self.res_6 = ResidualBlock(filters=filters, dilation_rate=1)
        # Conv3x3 for output
        self.conv = Conv3x3(filters=1)

    def call(self, inputs, training=None, mask=None):
        # inputs: [low disparity, resized left image], [(N, H, W, 1), ([N, H, W, 1)])
        assert len(inputs) == 2

        # Up-sample low disparity to corresponding resolution as the resized image
        scale_factor = inputs[1].shape[1] / inputs[0].shape[1]
        disparity = tf.image.resize(images=inputs[0], size=[inputs[1].shape[1], inputs[1].shape[2]])
        disparity = disparity * scale_factor

        # Concat the disparity and image
        concat = tf.concat([disparity, inputs[1]], -1)
        # learn disparity residual signal
        outputs = self.conv3x3(concat)
        outputs = self.res_1(outputs)
        outputs = self.res_2(outputs)
        outputs = self.res_3(outputs)
        outputs = self.res_4(outputs)
        outputs = self.res_5(outputs)
        outputs = self.res_6(outputs)
        outputs = self.conv(outputs)
        outputs += disparity

        return outputs     # refined disparity, (N, H, W, 1)
