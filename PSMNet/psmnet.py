import os
import glob
import time
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from component import *
from data_reader import read_image
from evaluation import evaluate_all


class PSMNet:
    def __init__(self, height, width, channel, min_disp, max_disp):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def build_model(self):
        # inputs
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        # CNN + SPP module
        extractor = FeatureExtractor(filters=32)
        left_feature = extractor(left_image)
        right_feature = extractor(right_image)

        # cost volume
        constructor = CostVolume(self.min_disp // 4, self.max_disp // 4)
        cost_volume = constructor([left_feature, right_feature])

        # 3D CNN (stacked hourglass)
        hourglass = StackedHourglass(filters=32)
        [out1, out2, out3] = hourglass(cost_volume)

        # disparity
        estimation = Estimation(self.min_disp, self.max_disp)
        d1 = estimation(out1)
        d2 = estimation(out2)
        d3 = estimation(out3)

        # build model
        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d1, d2, d3])
        self.model.summary()

    def predict(self, left_dir, right_dir, output_dir, weights):
        self.model.load_weights(weights)
        lefts = os.listdir(left_dir)
        rights = os.listdir(right_dir)
        lefts.sort()
        rights.sort()
        assert len(lefts) == len(rights)
        t0 = time.time()
        for left, right in zip(lefts, rights):
            left_image = np.expand_dims(read_image(os.path.join(left_dir, left)), 0)
            right_image = np.expand_dims(read_image(os.path.join(right_dir, right)), 0)
            disparity = self.model.predict([left_image, right_image])
            disparity = Image.fromarray(disparity[-1][0, :, :, 0])
            name = left.replace('left', 'disparity')
            disparity.save(os.path.join(output_dir, name))
        t1 = time.time()
        print('Total time: ', t1 - t0)


if __name__ == '__main__':
    # # predict
    # left_dir = 'the directory of left images'
    # right_dir = 'the directory of right images'
    # output_dir = 'the directory to save results'
    # weights = 'the weight file'
    # net = PSMNet(1024, 1024, 1, -128.0, 64.0)
    # net.build_model()
    # net.predict(left_dir, right_dir, output_dir, weights)

    # # evaluation
    # gt_dir = 'the directory of ground truth labels'
    # evaluate_all(output_dir, gt_dir, -128.0, 64.0)

    pass
