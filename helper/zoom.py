import os
import helper.webuiapi as webuiapi
from PIL import Image, ImageDraw
import cv2
import numpy as np


def zoom_ratio_detect(w, h, zoom_area_limit, zoom_max_resolusion):
    area = w * h
    scale = zoom_area_limit / area
    scale = scale ** (1 / 2)

    re_w = int(w * scale)
    re_h = int(h * scale)
    if re_w > zoom_max_resolusion:
        re_h = int(re_h * (zoom_max_resolusion / re_w))
        re_w = zoom_max_resolusion

    re_w = int(re_w / 8) * 8
    re_h = int(re_h / 8) * 8

    calc_w = w
    calc_h = int(re_h * (w / re_w))

    return [re_w, re_h, calc_w, calc_h]


def blur_masks(masks, dilation_factor, iter=1):
    dilated_masks = []
    if dilation_factor == 0:
        return masks
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        dilated_mask = cv2.erode(cv2_mask, kernel, iter)
        dilated_mask = cv2.GaussianBlur(dilated_mask, (51, 51), 0)

        dilated_masks.append(Image.fromarray(dilated_mask))
    return dilated_masks


def process(frame_index, input_img_arr, zoom_rects, zoom_area_limit, zoom_max_resolusion, zoom_image_folder, output_filename):
    zoom_image_list = []
    zoom_coords = []
    masks = []
    (input_img_height, input_img_width) = input_img_arr.shape[:2]
    for zoom_index, zoom_rect in enumerate(zoom_rects):
        print(zoom_rect)
        [x, y, w, h, start_frame, end_frame] = zoom_rect
        if frame_index + 1 >= start_frame and frame_index + 1 <= end_frame:
            [re_w, re_h, calc_w, calc_h] = zoom_ratio_detect(w, h, zoom_area_limit, zoom_max_resolusion)
            print([re_w, re_h, calc_w, calc_h])
            zoom_coords.append([x, y, re_w, re_h, calc_w, calc_h])

            mask = Image.new("L", [input_img_width, input_img_height], 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle([x, y, x + calc_w, y + calc_h], fill=255)
            masks.append(mask)

            img_array = input_img_arr
            zoom_img = Image.fromarray(img_array[y : y + calc_h, x : x + calc_w])
            zoom_image_list.append(zoom_img)

    masks = blur_masks(masks, 25)
    for mask_index, mask in enumerate(masks):
        output_mask_filename = f"{os.path.splitext(output_filename)[0]}-zoom-mask{mask_index}.png"
        output_mask_image_path = os.path.join(zoom_image_folder, output_mask_filename)
        mask.save(output_mask_image_path)

    return (zoom_rects, zoom_image_list, zoom_coords, masks)
