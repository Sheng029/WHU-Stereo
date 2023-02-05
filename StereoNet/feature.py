import tensorflow.keras as keras


L2 = 1.0e-4
alpha = 0.2


class Conv3x3(keras.Model):
    """
    Conv3x3 without bn or activation.
    """
    def __init__(self, filters):
        super(Conv3x3, self).__init__()
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                                        activation=None, use_bias=True, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv(inputs)

        return outputs


class Conv5x5BnRelu(keras.Model):
    """
    Conv5x5 followed by bn and relu.
    """
    def __init__(self, filters, strides):
        super(Conv5x5BnRelu, self).__init__()
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=5, strides=strides, padding='same',
                                        activation=None, use_bias=False, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        return outputs


class ResidualBlock(keras.Model):
    """
    Residual block, conv3x3, bn, leaky relu.
    """
    def __init__(self, filters, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                                         dilation_rate=dilation_rate, activation=None,
                                         use_bias=False, kernel_initializer='he_normal',
                                         kernel_regularizer=keras.regularizers.l2(L2))
        self.bn1 = keras.layers.BatchNormalization()
        self.leaky_relu1 = keras.layers.LeakyReLU(alpha=alpha)

        self.conv2 = keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                                   dilation_rate=dilation_rate, activation=None,
                                   use_bias=False, kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(L2))
        self.bn2 = keras.layers.BatchNormalization()

        self.leaky_relu2 = keras.layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.leaky_relu1(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        outputs += inputs
        outputs = self.leaky_relu2(outputs)

        return outputs


class FeatureExtractor(keras.Model):
    """
    Feature extractor used in StereoNet,
    down-sampling the input to its 1/8.
    """
    def __init__(self, filters):
        super(FeatureExtractor, self).__init__()
        # Three Conv5x5 (with a stride of 2) followed by bn and relu
        self.conv5x5_1 = Conv5x5BnRelu(filters=filters, strides=2)
        self.conv5x5_2 = Conv5x5BnRelu(filters=filters, strides=2)
        self.conv5x5_3 = Conv5x5BnRelu(filters=filters, strides=2)

        # 6 residual blocks that employ conv3x3, bn, and leaky relu
        self.res_1 = ResidualBlock(filters=filters, dilation_rate=1)
        self.res_2 = ResidualBlock(filters=filters, dilation_rate=1)
        self.res_3 = ResidualBlock(filters=filters, dilation_rate=1)
        self.res_4 = ResidualBlock(filters=filters, dilation_rate=1)
        self.res_5 = ResidualBlock(filters=filters, dilation_rate=1)
        self.res_6 = ResidualBlock(filters=filters, dilation_rate=1)

        # Conv3x3 without bn or activation
        self.conv3x3 = Conv3x3(filters=filters)

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv5x5_1(inputs)
        outputs = self.conv5x5_2(outputs)
        outputs = self.conv5x5_3(outputs)

        outputs = self.res_1(outputs)
        outputs = self.res_2(outputs)
        outputs = self.res_3(outputs)
        outputs = self.res_4(outputs)
        outputs = self.res_5(outputs)
        outputs = self.res_6(outputs)

        outputs = self.conv3x3(outputs)

        return outputs     # [N, H, W, C]
