import tensorflow.keras as keras


L2 = 1.0e-5


def conv2d(filters, kernel_size, strides, padding, dilation_rate):
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               dilation_rate=dilation_rate, use_bias=True,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))


def conv2d_bn(filters, kernel_size, strides, padding, dilation_rate, activation):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               dilation_rate=dilation_rate, use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential([conv, bn, relu])
    else:
        return keras.Sequential([conv, bn])


def avg_pool(pool_size, filters):
    pool = keras.layers.AvgPool2D(pool_size=pool_size)
    conv = keras.layers.Conv2D(filters, kernel_size=1, strides=1,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))

    return keras.Sequential([pool, conv])


class BasicBlock(keras.Model):
    def __init__(self, filters, dilation_rate):
        super(BasicBlock, self).__init__()

        self.conv1 = conv2d_bn(filters, 3, 1, 'same', dilation_rate, True)
        self.conv2 = conv2d_bn(filters, 3, 1, 'same', dilation_rate, False)
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        x = self.relu(x)

        return x


def make_blocks(filters, dilation_rate, num):
    blocks = keras.Sequential()
    for i in range(num):
        blocks.add(BasicBlock(filters, dilation_rate))

    return blocks


class FeatureExtraction(keras.Model):
    def __init__(self, filters):
        super(FeatureExtraction, self).__init__()

        self.conv0_1 = conv2d_bn(filters, 5, 2, 'same', 1, True)
        self.conv0_2 = conv2d_bn(2 * filters, 5, 2, 'same', 1, True)

        self.conv1_0 = make_blocks(2 * filters, 1, 4)
        self.conv1_1 = make_blocks(2 * filters, 2, 2)
        self.conv1_2 = make_blocks(2 * filters, 4, 2)
        self.conv1_3 = make_blocks(2 * filters, 1, 2)

        self.branch0 = avg_pool(1, filters)
        self.branch1 = avg_pool(2, filters)
        self.branch2 = avg_pool(4, filters)

    def call(self, inputs, training=None, mask=None):
        x = self.conv0_1(inputs)
        x = self.conv0_2(x)

        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        return [x0, x1, x2]   # [1/4, 1/8, 1/16]
