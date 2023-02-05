import tensorflow as tf
from submodule import *


class FeatureExtractor(keras.Model):
    """
    Feature extraction
    """
    def __init__(self, filters):
        super(FeatureExtractor, self).__init__()

        self.conv0_1 = conv2d_bn(filters, 3, 2, 'same', 1, True)
        self.conv0_2 = conv2d_bn(filters, 3, 1, 'same', 1, True)
        self.conv0_3 = conv2d_bn(filters, 3, 1, 'same', 1, True)

        self.conv1_x = make_blocks(filters, 1, 2)

        self.conv2_1 = BasicBlockDown(filters, 1)
        self.conv2_x = make_blocks(2 * filters, 1, 12)

        self.conv3_x = make_blocks(2 * filters, 2, 2)

        self.conv4_x = make_blocks(2 * filters, 4, 2)

        self.branch_1 = avg_pool(64, 64, filters)
        self.branch_2 = avg_pool(32, 32, filters)
        self.branch_3 = avg_pool(16, 16, filters)
        self.branch_4 = avg_pool(8, 8, filters)

        self.fusion = keras.Sequential(layers=[
            conv2d_bn(filters, 3, 1, 'same', 1, True),
            conv2d(filters, 1, 1, 'valid', 1, False)])

    def call(self, inputs, training=None, mask=None):
        x = self.conv0_1(inputs)
        x = self.conv0_2(x)
        x = self.conv0_3(x)

        x1 = self.conv1_x(x)

        x2 = self.conv2_1(x1)
        x2 = self.conv2_x(x2)

        x3 = self.conv3_x(x2)

        x4 = self.conv4_x(x3)

        b1 = self.branch_1(x4)
        b1 = tf.image.resize(b1, [x4.shape[1], x4.shape[2]])

        b2 = self.branch_2(x4)
        b2 = tf.image.resize(b2, [x4.shape[1], x4.shape[2]])

        b3 = self.branch_3(x4)
        b3 = tf.image.resize(b3, [x4.shape[1], x4.shape[2]])

        b4 = self.branch_4(x4)
        b4 = tf.image.resize(b4, [x4.shape[1], x4.shape[2]])

        con = tf.concat([x2, x4, b1, b2, b3, b4], -1)

        fusion = self.fusion(con)

        return fusion


class CostVolume(keras.Model):
    """
    Concat left and right feature vectors to form a cost volume.
    The cost volume covers both negative and positive disparities.
    """
    def __init__(self, min_disp, max_disp):
        super(CostVolume, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def call(self, inputs, training=None, mask=None):
        assert len(inputs) == 2
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                cost_volume.append(tf.pad(
                    tensor=tf.concat([inputs[0][:, :, :i, :], inputs[1][:, :, -i:, :]], -1),
                    paddings=[[0, 0], [0, 0], [0, -i], [0, 0]], mode='CONSTANT'))
            elif i > 0:
                cost_volume.append(tf.pad(
                    tensor=tf.concat([inputs[0][:, :, i:, :], inputs[1][:, :, :-i, :]], -1),
                    paddings=[[0, 0], [0, 0], [i, 0], [0, 0]], mode='CONSTANT'))
            else:
                cost_volume.append(tf.concat([inputs[0], inputs[1]], -1))
        cost_volume = tf.stack(cost_volume, 1)
        return cost_volume


class StackedHourglass(keras.Model):
    """
    Hourglass module.
    """
    def __init__(self, filters):
        super(StackedHourglass, self).__init__()

        self.conv0_1 = keras.Sequential(layers=[conv3d_bn(filters // 2, 3, 1, 'same', True),
                                                conv3d_bn(filters // 2, 3, 1, 'same', True)])
        self.conv0_2 = keras.Sequential(layers=[conv3d_bn(filters // 2, 3, 1, 'same', True),
                                                conv3d_bn(filters // 2, 3, 1, 'same', False)])

        self.conv1 = conv3d_bn(filters, 3, 2, 'same', True)
        self.conv2 = conv3d_bn(filters, 3, 1, 'same', False)
        self.conv3 = conv3d_bn(filters, 3, 2, 'same', True)
        self.conv4 = conv3d_bn(filters, 3, 1, 'same', True)
        self.conv5 = deconv3d_bn(filters, 3, 2, 'same', False)
        self.conv6 = deconv3d_bn(filters // 2, 3, 2, 'same', False)

        self.conv7 = conv3d_bn(filters, 3, 2, 'same', True)
        self.conv8 = conv3d_bn(filters, 3, 1, 'same', False)
        self.conv9 = conv3d_bn(filters, 3, 2, 'same', True)
        self.conv10 = conv3d_bn(filters, 3, 1, 'same', True)
        self.conv11 = deconv3d_bn(filters, 3, 2, 'same', False)
        self.conv12 = deconv3d_bn(filters // 2, 3, 2, 'same', False)

        self.conv13 = conv3d_bn(filters, 3, 2, 'same', True)
        self.conv14 = conv3d_bn(filters, 3, 1, 'same', False)
        self.conv15 = conv3d_bn(filters, 3, 2, 'same', True)
        self.conv16 = conv3d_bn(filters, 3, 1, 'same', True)
        self.conv17 = deconv3d_bn(filters, 3, 2, 'same', False)
        self.conv18 = deconv3d_bn(filters // 2, 3, 2, 'same', False)

        self.out1 = keras.Sequential(layers=[conv3d_bn(filters // 2, 3, 1, 'same', True),
                                             conv3d(1, 3, 1, 'same', False)])
        self.out2 = keras.Sequential(layers=[conv3d_bn(filters // 2, 3, 1, 'same', True),
                                             conv3d(1, 3, 1, 'same', False)])
        self.out3 = keras.Sequential(layers=[conv3d_bn(filters // 2, 3, 1, 'same', True),
                                             conv3d(1, 3, 1, 'same', False)])

        self.upsample1 = keras.layers.UpSampling3D(size=(4, 4, 4))
        self.upsample2 = keras.layers.UpSampling3D(size=(4, 4, 4))
        self.upsample3 = keras.layers.UpSampling3D(size=(4, 4, 4))

    def call(self, inputs, training=None, mask=None):
        # inputs: [N, D, H, W, C]
        x0 = self.conv0_1(inputs)
        x1 = self.conv0_2(x0)
        x1 += x0

        x2 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x2)
        x3 = self.conv4(x3)
        x4 = self.conv5(x3)
        x4 += x2
        x5 = self.conv6(x4)
        x5 += x1

        x6 = self.conv7(x5)
        x6 = self.conv8(x6)
        x6 += x4
        x7 = self.conv9(x6)
        x7 = self.conv10(x7)
        x8 = self.conv11(x7)
        x8 += x2
        x9 = self.conv12(x8)
        x9 += x1

        x10 = self.conv13(x9)
        x10 = self.conv14(x10)
        x10 += x8
        x11 = self.conv15(x10)
        x11 = self.conv16(x11)
        x12 = self.conv17(x11)
        x12 += x2
        x13 = self.conv18(x12)
        x13 += x1

        # return [x5, x9, x13]
        outputs_1 = self.out1(x5)      # [N, D, H, W, 1], 1/4
        outputs_2 = self.out2(x9)      # [N, D, H, W, 1], 1/4
        outputs_2 += outputs_1
        outputs_3 = self.out3(x13)     # [N, D, H, W, 1], 1/4
        outputs_3 += outputs_2

        outputs_1 = self.upsample1(outputs_1)      # [N, D, H, W, 1]
        outputs_2 = self.upsample2(outputs_2)      # [N, D, H, W, 1]
        outputs_3 = self.upsample3(outputs_3)      # [N, D, H, W, 1]

        outputs_1 = tf.transpose(tf.squeeze(outputs_1, -1), (0, 2, 3, 1))    # [N, H, W, D]
        outputs_2 = tf.transpose(tf.squeeze(outputs_2, -1), (0, 2, 3, 1))    # [N, H, W, D]
        outputs_3 = tf.transpose(tf.squeeze(outputs_3, -1), (0, 2, 3, 1))    # [N, H, W, D]

        return [outputs_1, outputs_2, outputs_3]


class Estimation(keras.Model):
    """
    Soft ArgMin.
    """
    def __init__(self, min_disp, max_disp):
        super(Estimation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def call(self, inputs, training=None, mask=None):
        assert inputs.shape[-1] == self.max_disp - self.min_disp
        candidates = tf.linspace(1.0 * self.min_disp, 1.0 * self.max_disp - 1.0, self.max_disp - self.min_disp)
        probabilities = tf.math.softmax(-1.0 * inputs, -1)
        disparities = tf.reduce_sum(candidates * probabilities, -1, True)
        return disparities
