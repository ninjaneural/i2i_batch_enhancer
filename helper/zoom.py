import os
import helper.webuiapi as webuiapi
from PIL import Image, ImageDraw
from helper.image_util import blur_masks
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


def process(frame_index, input_img_arr, zoom_rects, zoom_blur, zoom_area_limit, zoom_max_resolusion, zoom_image_folder, output_filename):
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
            masks.append(np.array(mask))

            img_array = input_img_arr
            zoom_img = Image.fromarray(img_array[y : y + calc_h, x : x + calc_w])
            zoom_image_list.append(zoom_img)

    mask_arrs = blur_masks(masks, zoom_blur)
    for mask_index, mask_arr in enumerate(mask_arrs):
        mask = Image.fromarray(mask_arr)
        output_mask_filename = f"{os.path.splitext(output_filename)[0]}-zoom-mask{mask_index}.png"
        output_mask_image_path = os.path.join(zoom_image_folder, output_mask_filename)
        mask.save(output_mask_image_path)

    return (zoom_image_list, zoom_coords, mask_arrs)
