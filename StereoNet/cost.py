import tensorflow as tf
import tensorflow.keras as keras


class Difference(keras.Model):
    def __init__(self, min_disp, max_disp):
        super(Difference, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def call(self, inputs, training=None, mask=None):
        assert len(inputs) == 2
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                cost_volume.append(tf.pad(
                    tensor=inputs[0][:, :, :i, :] - inputs[1][:, :, -i:, :],
                    paddings=[[0, 0], [0, 0], [0, -i], [0, 0]], mode='CONSTANT'))
            elif i > 0:
                cost_volume.append(tf.pad(
                    tensor=inputs[0][:, :, i:, :] - inputs[1][:, :, :-i, :],
                    paddings=[[0, 0], [0, 0], [i, 0], [0, 0]], mode='CONSTANT'))
            else:
                cost_volume.append(inputs[0] - inputs[1])
        cost_volume = tf.stack(cost_volume, 1)
        return cost_volume
