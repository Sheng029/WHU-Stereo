import os
import glob
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from feature import FeatureExtractor
from cost import Difference
from aggregation import CostAggregation
from refinement import Refinement
from computation import Computation
from data_reader import read_image
from evaluation import evaluate_all


class StereoNet:
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

        # extract features
        extractor = FeatureExtractor(filters=32)
        left_feature = extractor(left_image)
        right_feature = extractor(right_image)

        # construct cost volume
        constructor = Difference(self.min_disp // 8, self.max_disp // 8)
        cost_volume = constructor([left_feature, right_feature])

        # cost aggregation
        aggregator = CostAggregation(filters=32)
        cost_volume = aggregator(cost_volume)

        # pre-refined disparity
        computer = Computation(self.min_disp // 8, self.max_disp // 8)
        d0 = computer(cost_volume)

        # hierarchical refinement
        refiner1 = Refinement(filters=32)  # [None, 256, 256, 1], 1/4 resolution
        left_image_4x = tf.image.resize(left_image, [self.height // 4, self.width // 4])
        d1 = refiner1([d0, left_image_4x])

        refiner2 = Refinement(filters=32)  # [None, 512, 512, 1], 1/2 resolution
        left_image_2x = tf.image.resize(left_image, [self.height // 2, self.width // 2])
        d2 = refiner2([d1, left_image_2x])

        refiner3 = Refinement(filters=32)  # [None, 1024, 1024, 1], full resolution
        d3 = refiner3([d2, left_image])

        # The predicted disparity map is always bilinearly up-sampled to match ground-truth resolution.
        d0 = tf.image.resize(d0, [self.height, self.width]) * 8
        d1 = tf.image.resize(d1, [self.height, self.width]) * 4
        d2 = tf.image.resize(d2, [self.height, self.width]) * 2

        # StereoNet model
        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d0, d1, d2, d3])
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
    # net = StereoNet(1024, 1024, 1, -128.0, 64.0)
    # net.build_model()
    # net.predict(left_dir, right_dir, output_dir, weights)

    # # evaluation
    # gt_dir = 'the directory of ground truth labels'
    # evaluate_all(output_dir, gt_dir, -128.0, 64.0)

    pass
