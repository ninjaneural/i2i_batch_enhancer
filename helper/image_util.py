import cv2


def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation = cv2.INTER_CUBIC
    else:
        interpolation = interpolation = cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)


def img_crop_and_resize(img_array, coords, re_w, re_h):
    (x, y, w, h) = coords
    crop_image = img_array[y : y + h, x : x + w]
    return resize_img(crop_image, re_w, re_h)
