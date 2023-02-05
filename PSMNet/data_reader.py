import cv2
import random
import numpy as np


def read_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    image_norm = (image - np.mean(image)) / np.std(image)
    return np.expand_dims(image_norm.astype('float32'), -1)


def read_disp(filename):
    disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return np.expand_dims(disp, -1)


def read_batch(left_paths, right_paths, disp_paths):
    left_images, right_images, disp_maps = [], [], []
    for left_path, right_path, disp_path in zip(left_paths, right_paths, disp_paths):
        left_images.append(read_image(left_path))
        right_images.append(read_image(right_path))
        disp_maps.append(read_disp(disp_path))
    return np.array(left_images), np.array(right_images), np.array(disp_maps)


def load_batch(all_left_paths, all_right_paths, all_disp_paths, batch_size=4, reshuffle=False):
    assert len(all_left_paths) == len(all_disp_paths)
    assert len(all_right_paths) == len(all_disp_paths)
    i = 0
    while True:
        left_images, right_images, disp_maps = read_batch(
            left_paths=all_left_paths[i * batch_size:(i + 1) * batch_size],
            right_paths=all_right_paths[i * batch_size:(i + 1) * batch_size],
            disp_paths=all_disp_paths[i * batch_size:(i + 1) * batch_size])
        yield [left_images, right_images], [disp_maps, disp_maps, disp_maps]
        i = (i + 1) % (len(all_left_paths) // batch_size)
        if reshuffle:
            if i == 0:
                paths = list(zip(all_left_paths, all_right_paths, all_disp_paths))
                random.shuffle(paths)
                all_left_paths, all_right_paths, all_disp_paths = zip(*paths)
