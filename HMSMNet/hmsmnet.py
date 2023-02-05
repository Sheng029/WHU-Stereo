import os
import glob
import time
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from feature import FeatureExtraction
from cost import CostConcatenation
from aggregation import Hourglass, FeatureFusion
from computation import Estimation
from refinement import Refinement
from data_reader import read_left, read_right
from evaluation import evaluate_all


class HMSMNet:
    def __init__(self, height, width, channel, min_disp, max_disp):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def build_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))
        gx = keras.Input(shape=(self.height, self.width, self.channel))
        gy = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extraction = FeatureExtraction(filters=16)
        [l0, l1, l2] = feature_extraction(left_image)
        [r0, r1, r2] = feature_extraction(right_image)

        cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        cost2 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        cost_volume0 = cost0([l0, r0])
        cost_volume1 = cost1([l1, r1])
        cost_volume2 = cost2([l2, r2])

        hourglass0 = Hourglass(filters=16)
        hourglass1 = Hourglass(filters=16)
        hourglass2 = Hourglass(filters=16)
        agg_cost0 = hourglass0(cost_volume0)
        agg_cost1 = hourglass1(cost_volume1)
        agg_cost2 = hourglass2(cost_volume2)

        estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        disparity2 = estimator2(agg_cost2)

        fusion1 = FeatureFusion(units=16)
        fusion_cost1 = fusion1([agg_cost2, agg_cost1])
        hourglass3 = Hourglass(filters=16)
        agg_fusion_cost1 = hourglass3(fusion_cost1)

        estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        disparity1 = estimator1(agg_fusion_cost1)

        fusion2 = FeatureFusion(units=16)
        fusion_cost2 = fusion2([agg_fusion_cost1, agg_cost0])
        hourglass4 = Hourglass(filters=16)
        agg_fusion_cost2 = hourglass4(fusion_cost2)

        estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        disparity0 = estimator0(agg_fusion_cost2)

        # refinement
        refiner = Refinement(filters=32)
        final_disp = refiner([disparity0, left_image, gx, gy])

        self.model = keras.Model(inputs=[left_image, right_image, gx, gy],
                                 outputs=[disparity2, disparity1, disparity0, final_disp])
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
            left_image, gx, gy = read_left(os.path.join(left_dir, left))
            right_image = read_right(os.path.join(right_dir, right))
            left_image = np.expand_dims(left_image, 0)
            gx = np.expand_dims(gx, 0)
            gy = np.expand_dims(gy, 0)
            right_image = np.expand_dims(right_image, 0)
            disparity = self.model.predict([left_image, right_image, gx, gy])
            disparity = disparity[-1][0, :, :, 0]
            disparity = Image.fromarray(disparity)
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
    # net = HMSMNet(1024, 1024, 1, -128.0, 64.0)
    # net.build_model()
    # net.predict(left_dir, right_dir, output_dir, weights)

    # # evaluation
    # gt_dir = 'the directory of ground truth labels'
    # evaluate_all(output_dir, gt_dir, -128.0, 64.0)

    pass
