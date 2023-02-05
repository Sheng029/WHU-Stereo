import tensorflow as tf
import tensorflow.keras as keras


L2 = 1.0e-4
alpha = 0.2


class Conv3D(keras.Model):
    """
    Conv3D (kernel 3x3x3) without bn or activation.
    """

    def __init__(self, filters):
        super(Conv3D, self).__init__()
        self.conv3d = keras.layers.Conv3D(filters=filters, kernel_size=3, strides=1, padding='same',
                                          activation=None, use_bias=True, kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(L2))

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv3d(inputs)

        return outputs


class Conv3DBnLeakyRelu(keras.Model):
    """
    Conv3D (kernel 3x3x3) followed by bn and leaky relu.
    """

    def __init__(self, filters):
        super(Conv3DBnLeakyRelu, self).__init__()
        self.conv3d = keras.layers.Conv3D(filters=filters, kernel_size=3, strides=1, padding='same',
                                          activation=None, use_bias=False, kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(L2))
        self.bn = keras.layers.BatchNormalization()
        self.leaky_relu = keras.layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv3d(inputs)
        outputs = self.bn(outputs)
        outputs = self.leaky_relu(outputs)

        return outputs


class CostAggregation(keras.Model):
    """
    Cost aggregation used in StereoNet.
    """

    def __init__(self, filters):
        super(CostAggregation, self).__init__()

        # 4 conv3d followed by bn and leaky relu
        self.conv3d_1 = Conv3DBnLeakyRelu(filters=filters)
        self.conv3d_2 = Conv3DBnLeakyRelu(filters=filters)
        self.conv3d_3 = Conv3DBnLeakyRelu(filters=filters)
        self.conv3d_4 = Conv3DBnLeakyRelu(filters=filters)

        # final conv3d without bn or activation
        self.conv3d = Conv3D(filters=1)

    def call(self, inputs, training=None, mask=None):
        # inputs shape: [N, D, H, W, C]
        assert len(inputs.shape) == 5

        outputs = self.conv3d_1(inputs)
        outputs = self.conv3d_2(outputs)
        outputs = self.conv3d_3(outputs)
        outputs = self.conv3d_4(outputs)

        outputs = self.conv3d(outputs)  # [N, D, H, W, 1]

        outputs = tf.squeeze(outputs, -1)  # [N, D, H, W]
        outputs = tf.transpose(outputs, (0, 2, 3, 1))  # [N, H, W, D]

        return outputs

