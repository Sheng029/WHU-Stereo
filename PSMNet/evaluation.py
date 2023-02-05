import glob
import cv2
import numpy as np


def compute_epe(est_path, gt_path, min_disp, max_disp):
    est = cv2.imread(est_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    zeros = np.zeros_like(gt, 'int32')
    ones = np.ones_like(gt, 'int32')
    mask1 = np.where(gt >= max_disp, zeros, ones)
    mask2 = np.where(gt < min_disp, zeros, ones)
    mask = mask1 & mask2

    error = np.sum(np.abs(est - gt) * mask)
    nums = np.sum(mask)
    epe = error / nums

    return error, nums, epe


def compute_d1(est_path, gt_path, min_disp, max_disp):
    est = cv2.imread(est_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    zeros = np.zeros_like(gt, 'int32')
    ones = np.ones_like(gt, 'int32')
    mask1 = np.where(gt >= max_disp, zeros, ones)
    mask2 = np.where(gt < min_disp, zeros, ones)
    mask = mask1 & mask2

    err_map = np.abs(est - gt) * mask
    err_mask = err_map > 3
    err_disps = np.sum(err_mask.astype('float32'))
    nums = np.sum(mask)
    d1 = err_disps / nums

    return err_disps, nums, d1


def evaluate(est_path, gt_path, min_disp, max_disp):
    error, nums, epe = compute_epe(est_path, gt_path, min_disp, max_disp)
    print('Sum of absolute error: %f, num of valid pixels: %d, end-point-error: %f'
          % (error, int(nums), epe))
    err_disps, nums, d1 = compute_d1(est_path, gt_path, min_disp, max_disp)
    print('Num of error disparities: %d, num of valid pixels: %d, d1: %f'
          % (int(err_disps), int(nums), d1))


def evaluate_all(est_dir, gt_dir, min_disp, max_disp):
    est_paths = glob.glob(est_dir + '/*')
    gt_paths = glob.glob(gt_dir + '/*')
    est_paths.sort()
    gt_paths.sort()
    assert len(est_paths) == len(gt_paths)

    total_error, total_nums = 0, 0
    for est_path, gt_path in zip(est_paths, gt_paths):
        error, nums, epe = compute_epe(est_path, gt_path, min_disp, max_disp)
        total_error += error
        total_nums += nums
    print('\nEnd-point-error: %f' % (total_error / total_nums))

    total_err_disps, total_nums = 0, 0
    for est_path, gt_path in zip(est_paths, gt_paths):
        err_disps, nums, d1 = compute_d1(est_path, gt_path, min_disp, max_disp)
        total_err_disps += err_disps
        total_nums += nums
    print('\nD1: %f' % (total_err_disps / total_nums))
