import os
import glob
from PIL import Image, ImageDraw
import numpy as np
from helper.config import Config
from helper.image_util import blur_masks, resize_image, merge_image
from helper.util import get_image_paths


def run(config: Config, project_folder: str, patch_name: str, patch_rects: list):
    if project_folder:
        output_folder = os.path.normpath(os.path.join(project_folder, config.output_folder))
        input_folder = os.path.normpath(os.path.join(project_folder, config.input_folder))
        patch_output_folder = os.path.normpath(os.path.join(project_folder, "./" + patch_name))
        patch_mask_output_folder = os.path.normpath(os.path.join(project_folder, "./" + patch_name + "/mask"))
    else:
        output_folder = config.output_folder
        input_folder = config.input_folder
        patch_output_folder = project_folder, "./" + patch_name
        patch_mask_output_folder = project_folder, "./" + patch_name + "/mask"

    os.makedirs(patch_output_folder, exist_ok=True)
    os.makedirs(patch_mask_output_folder, exist_ok=True)

    bg_images_path_list = get_image_paths(output_folder)
    particle_images_path_list = get_image_paths(input_folder)

    frame_width = config.frame_width
    frame_height = config.frame_height

    mod_y = 0

    for i in range(0, len(bg_images_path_list)):
        print(f"frame:{i+1}")

        bg_img = None
        particle_img = None
        output_filename = os.path.basename(bg_images_path_list[i])

        patch_coords = []
        masks = []
        for patch_index, patch_rect in enumerate(patch_rects):
            # print(patch_rect)
            [x, y, w, h, start_frame, end_frame] = patch_rect
            if i + 1 >= start_frame and i + 1 <= end_frame:
                if bg_img == None:
                    bg_img = Image.open(bg_images_path_list[i])
                    particle_img = Image.open(particle_images_path_list[i])

                    bg_img_arr = np.array(bg_img)
                    particle_img_arr = np.array(particle_img)

                    if bg_img.width != frame_width or bg_img.height != frame_height:
                        fit_height = (int)(bg_img.height * (frame_width / bg_img.width))
                        if fit_height != bg_img.height:
                            bg_img_arr = resize_image(bg_img_arr, frame_width, fit_height)
                        if frame_height > fit_height:
                            top = (int)((frame_height - fit_height) / 2) + mod_y
                            bottom = (frame_height - fit_height) - top
                            bg_img_arr = np.pad(bg_img_arr, pad_width=((top, bottom), (0, 0), (0, 0)), mode="constant")
                        elif fit_height > frame_height:
                            top = (int)((fit_height - frame_height) / 2)
                            bottom = (fit_height - frame_height) - top
                            bg_img_arr = bg_img_arr[top:frame_height, 0:frame_width]
                        output_bg_filename = f"{os.path.splitext(output_filename)[0]}-cover.png"
                        Image.fromarray(bg_img_arr).save(os.path.join(patch_mask_output_folder, output_bg_filename))

                    if particle_img.width != frame_width or particle_img.height != frame_height:
                        particle_img_arr = resize_image(particle_img_arr, frame_width, frame_height)

                mask = Image.new("L", [frame_width, frame_height], 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([x, y + mod_y, x + w, y + mod_y + h], fill=255)
                masks.append(mask)
                patch_coords.append([x, y + mod_y, w, h])

        masks = blur_masks(masks, 20)
        for mask_index, mask in enumerate(masks):
            output_mask_filename = f"{os.path.splitext(output_filename)[0]}-{mask_index}.png"
            output_mask_image_path = os.path.join(patch_mask_output_folder, output_mask_filename)
            mask.save(output_mask_image_path)

        for patch_index, (patch_coord, masks) in enumerate(zip(patch_coords, masks)):
            [x, y, w, h] = patch_coord
            particle_img_arr = merge_image(bg_img_arr, particle_img_arr, patch_coord, mask)

        if len(patch_coords) >= 1:
            merged_image = Image.fromarray(particle_img_arr)
            merged_image.save(os.path.join(patch_output_folder, output_filename))

            print(f"patch image for frame {i}:")
