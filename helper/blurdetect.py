import numpy as np
import blur_detector
import cv2
from PIL import Image
from helper.image_util import resize_image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def blur_merge(orig_image, convert_image, coord):
    blur_image = rgb2gray(orig_image)
    print(blur_image)
    blur_map = blur_detector.detectBlur(
        blur_image, downsampling_factor=4, num_scales=4, scale_start=2, num_iterations_RF_filter=3, show_progress=False
    )
    print(blur_map)
    blur_map = blur_map * 2
    blur_map[blur_map > 1] = 1

    (x, y, w, h) = coord
    print(f"coord {coord}")

    print(f"convert_image {convert_image.shape}")
    convert_image = resize_image(convert_image, w, h, "crop")
    blur_map = blur_map[y : y + h, x : x + w]
    if blur_map.ndim == 2:
        blur_map = blur_map[:, :, np.newaxis]

    target_image = orig_image[y : y + h, x : x + w]
    print(f"target_image {target_image.shape}")
    print(f"blur_map {blur_map.shape}")
    print(f"convert_image {convert_image.shape}")

    convert_image = convert_image[0:h, 0:w]
    target_image[0:h, 0:w] = blur_map * convert_image + (1 - blur_map) * target_image
    print(f"target_image {target_image.shape}")
    return target_image


