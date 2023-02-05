import tensorflow.keras as keras


L2 = 1.0e-4


def conv2d_bn(filters, kernel_size, strides, padding, dilation_rate, activation=True):
    """
    2D convolution followed by bn (and relu).
    """
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, dilation_rate=dilation_rate, use_bias=False,
                               kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential(layers=[conv, bn, relu])
    else:
        return keras.Sequential(layers=[conv, bn])


def conv2d(filters, kernel_size, strides, padding, dilation_rate, use_bias=False):
    """
    2D convolution.
    """
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, dilation_rate=dilation_rate, use_bias=use_bias,
                               kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(L2))


def conv3d_bn(filters, kernel_size, strides, padding, activation=True):
    """
    3D convolution followed by bn (and relu).
    """
    conv = keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, use_bias=False, kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential(layers=[conv, bn, relu])
    else:
        return keras.Sequential(layers=[conv, bn])


def conv3d(filters, kernel_size, strides, padding, use_bias=False):
    """
    3D convolution.
    """
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, use_bias=use_bias,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))


def deconv3d_bn(filters, kernel_size, strides, padding, activation=True):
    """
    Conv3DTranspose, bn, (relu)
    """
    conv = keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, use_bias=False, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential(layers=[conv, bn, relu])
    else:
        return keras.Sequential(layers=[conv, bn])


def deconv3d(filters, kernel_size, strides, padding, use_bias=False):
    """
    Conv3DTranspose.
    """
    return keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, use_bias=use_bias, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))


def avg_pool(pool_size, strides, filters):
    """
    Average pooling, 1x1 convolution.
    """
    pool = keras.layers.AvgPool2D(pool_size, strides)
    conv = conv2d_bn(filters, 1, 1, 'valid', 1, True)

    return keras.Sequential(layers=[pool, conv])


class BasicBlock(keras.Model):
    """
    Residual block, no ReLU after summation.
    """
    def __init__(self, filters, dilation_rate):
        super(BasicBlock, self).__init__()

        self.conv_bn_1 = conv2d_bn(filters=filters, kernel_size=3, strides=1, padding='same',
                                   dilation_rate=dilation_rate, activation=True)
        self.conv_bn_2 = conv2d_bn(filters=filters, kernel_size=3, strides=1, padding='same',
                                   dilation_rate=dilation_rate, activation=False)

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv_bn_1(inputs)
        outputs = self.conv_bn_2(outputs)
        outputs += inputs

        return outputs


class BasicBlockDown(keras.Model):
    """
    Residual block, inputs are down-sampled.
    """
    def __init__(self, filters, dilation_rate):
        super(BasicBlockDown, self).__init__()

        self.conv_bn_1 = conv2d_bn(filters=filters * 2, kernel_size=3, strides=2, padding='same',
                                   dilation_rate=dilation_rate, activation=True)
        self.conv_bn_2 = conv2d_bn(filters=filters * 2, kernel_size=3, strides=1, padding='same',
                                   dilation_rate=dilation_rate, activation=False)
        self.conv_bn_id = conv2d_bn(filters=filters * 2, kernel_size=3, strides=2, padding='same',
                                    dilation_rate=dilation_rate, activation=False)

    def call(self, inputs, training=None, mask=None):
        outputs = self.conv_bn_1(inputs)
        outputs = self.conv_bn_2(outputs)
        identity = self.conv_bn_id(inputs)
        outputs += identity

        return outputs


def make_blocks(filters, dilation_rate, blocks):
    """
    Build residual block.
    """
    block = keras.Sequential()
    for i in range(blocks):
        block.add(layer=BasicBlock(filters, dilation_rate))

    return block
