import tensorflow as tf
import tensorflow.keras as keras


L2 = 1.0e-5
alpha = 0.2


def conv3d(filters, kernel_size, strides, padding):
    return keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))


def conv3d_bn(filters, kernel_size, strides, padding, activation):
    conv = keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               use_bias=False, kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    leaky_relu = keras.layers.LeakyReLU(alpha=alpha)

    if activation:
        return keras.Sequential([conv, bn, leaky_relu])
    else:
        return keras.Sequential([conv, bn])


def trans_conv3d_bn(filters, kernel_size, strides, padding, activation):
    conv = keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size,
                                        strides=strides, padding=padding,
                                        use_bias=False, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    leaky_relu = keras.layers.LeakyReLU(alpha=alpha)

    if activation:
        return keras.Sequential([conv, bn, leaky_relu])
    else:
        return keras.Sequential([conv, bn])


class Hourglass(keras.Model):
    def __init__(self, filters):
        super(Hourglass, self).__init__()

        self.conv1 = conv3d_bn(filters, 3, 1, 'same', True)
        self.conv2 = conv3d_bn(filters, 3, 1, 'same', True)
        self.conv3 = conv3d_bn(2 * filters, 3, 2, 'same', True)
        self.conv4 = conv3d_bn(2 * filters, 3, 1, 'same', True)
        self.conv5 = conv3d_bn(2 * filters, 3, 2, 'same', True)
        self.conv6 = conv3d_bn(2 * filters, 3, 1, 'same', True)
        self.conv7 = trans_conv3d_bn(2 * filters, 4, 2, 'same', True)
        self.conv8 = trans_conv3d_bn(filters, 4, 2, 'same', True)

    def call(self, inputs, training=None, mask=None):
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        x3 = self.conv5(x2)
        x3 = self.conv6(x3)
        x4 = self.conv7(x3)
        x4 += x2
        x5 = self.conv8(x4)
        x5 += x1

        return x5  # [N, D, H, W, C]


class FeatureFusion(keras.Model):
    def __init__(self, units):
        super(FeatureFusion, self).__init__()

        self.upsample = keras.layers.UpSampling3D(size=(2, 2, 2))
        self.add1 = keras.layers.Add()
        self.avg_pool3d = keras.layers.GlobalAvgPool3D()
        self.fc1 = keras.layers.Dense(units=units, use_bias=True,
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=keras.regularizers.l2(L2))
        self.relu = keras.layers.Activation(activation='relu')
        self.fc2 = keras.layers.Dense(units=units, use_bias=True,
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=keras.regularizers.l2(L2))
        self.sigmoid = keras.layers.Activation(activation='sigmoid')
        self.multiply1 = keras.layers.Multiply()
        self.multiply2 = keras.layers.Multiply()
        self.add2 = keras.layers.Add()

    def call(self, inputs, training=None, mask=None):
        # inputs[0]: lower, inputs[1]: higher
        assert len(inputs) == 2

        x1 = self.upsample(inputs[0])
        x2 = self.add1([x1, inputs[1]])
        shape = x2.get_shape().as_list()
        v = self.avg_pool3d(x2)
        v = self.fc1(v)
        v = self.relu(v)
        v = self.fc2(v)
        v = self.sigmoid(v)
        v1 = 1.0 - v
        v = tf.expand_dims(v, 1)
        v = tf.expand_dims(v, 1)
        v = tf.expand_dims(v, 1)
        v = tf.tile(v, [1, shape[1], shape[2], shape[3], 1])
        v1 = tf.expand_dims(v1, 1)
        v1 = tf.expand_dims(v1, 1)
        v1 = tf.expand_dims(v1, 1)
        v1 = tf.tile(v1, [1, shape[1], shape[2], shape[3], 1])
        x3 = self.multiply1([x1, v])
        x4 = self.multiply2([inputs[1], v1])
        x = self.add2([x3, x4])

        return x
