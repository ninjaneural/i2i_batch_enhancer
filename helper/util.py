import os
import glob
import numpy as np


def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def get_lora_paths(folder):
    image_extensions = "*.safetensors"
    files = []
    files.extend(glob.glob(os.path.join(folder, image_extensions)))
    return sorted(files)


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "valid")


def smooth_data(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

    # The "my_average" part: shrinks the averaging window on the side that
    # reaches beyond the data, keeps the other side the same size as given
    # by "span"
    re[0] = np.average(arr[:span])
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re


def intersect_rect(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        (0, 0, 0, 0)
    return (x, y, w, h)
