import os
import glob
from PIL import Image, ImageDraw
import cv2
import numpy as np

source_images_path = "./input"
target_images_path = "./output"
output_folder = "./overlap"
output_mask_folder = "./overlap/mask"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)


def blur_masks(masks, dilation_factor, iter=1, ksize=51):
    dilated_masks = []
    if dilation_factor == 0:
        return masks
    kernel = np.ones((dilation_factor, dilation_factor), np.uint8)
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        dilated_mask = cv2.erode(cv2_mask, kernel, iter)
        dilated_mask = cv2.GaussianBlur(dilated_mask, (ksize, ksize), 0)

        dilated_masks.append(Image.fromarray(dilated_mask))
    return dilated_masks


def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation = cv2.INTER_CUBIC
    else:
        interpolation = interpolation = cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)


def merge_face(source, target, rect, mask):
    [x, y, w, h] = rect
    mask_array = np.array(mask.convert("L"))
    mask_array = mask_array[y : y + h, x : x + w]
    mask_array = mask_array.astype(dtype="float") / 255
    # print(mask_array)
    if mask_array.ndim == 2:
        mask_array = mask_array[:, :, np.newaxis]

    bg = target[y : y + h, x : x + w]
    src = source[y : y + h, x : x + w]
    target[y : y + h, x : x + w] = mask_array * src + (1 - mask_array) * bg

    return target


def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


source_images_path_list = get_image_paths(source_images_path)
target_images_path_list = get_image_paths(target_images_path)


output_images = []
output_paths = []

prompt_index = 0

start_index = 115

patch_rects = []
patch_rects.append([267, 1405, 150, 150, 283, 290])


for i in range(start_index, len(source_images_path_list)):
    print(f"frame:{i+1}")
    # print(f"source image:{source_images_path_list[i]}")
    # print(f"target image:{target_images_path_list[i]}")

    source_img = None
    target_img = None
    output_filename = os.path.basename(source_images_path_list[i])

    patch_coords = []
    masks = []
    for patch_index, patch_rect in enumerate(patch_rects):
        # print(patch_rect)
        [x, y, w, h, start_frame, end_frame] = patch_rect
        if i + 1 >= start_frame and i + 1 <= end_frame:
            if source_img == None:
                source_img = Image.open(source_images_path_list[i])
                target_img = Image.open(target_images_path_list[i])

                source_img_arr = np.array(source_img)
                target_img_arr = np.array(target_img)

                frame_width = target_img.width
                frame_height = target_img.height

                if source_img.width != frame_width or source_img.height != frame_height:
                    source_img_arr = resize_img(source_img_arr, frame_width, frame_height)

            mask = Image.new("L", [frame_width, frame_height], 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle([x, y, x + w, y + h], fill=255)
            masks.append(mask)
            patch_coords.append([x, y, w, h])

    masks = blur_masks(masks, 20)
    for mask_index, mask in enumerate(masks):
        output_mask_filename = f"{os.path.splitext(output_filename)[0]}-{mask_index}.png"
        output_mask_image_path = os.path.join(output_mask_folder, output_mask_filename)
        mask.save(output_mask_image_path)

    for patch_index, (patch_coord, masks) in enumerate(zip(patch_coords, masks)):
        [x, y, w, h] = patch_coord
        target_img_arr = merge_face(source_img_arr, target_img_arr, patch_coord, mask)

    if len(patch_coords) >= 1:
        merge_image = Image.fromarray(target_img_arr)
        merge_image.save(os.path.join(output_folder, output_filename))

        print(f"Written data for frame {i}:")
