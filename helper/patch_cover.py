import os
import glob
from PIL import Image, ImageDraw
import numpy as np
from helper.config import Config, PatchConfig
from helper.image_util import blur_masks, resize_image, merge_image
from helper.facedetect import face_detect, process as face_process
from helper.util import get_image_paths


def run(config: Config, project_folder: str, patchconfig: PatchConfig):
    if project_folder:
        bg_folder = os.path.normpath(os.path.join(project_folder, patchconfig.bg_folder))
        particle_folder = os.path.normpath(os.path.join(project_folder, patchconfig.particle_folder))
        patch_output_folder = os.path.normpath(os.path.join(project_folder, patchconfig.output_folder))
        patch_mask_output_folder = os.path.normpath(os.path.join(project_folder, patchconfig.output_folder + "/mask"))
    else:
        bg_folder = patchconfig.bg_folder
        particle_folder = patchconfig.particle_folder
        patch_output_folder = patchconfig.output_folder
        patch_mask_output_folder = patchconfig.output_folder + "/mask"

    os.makedirs(patch_output_folder, exist_ok=True)
    os.makedirs(patch_mask_output_folder, exist_ok=True)

    bg_images_path_list = get_image_paths(bg_folder)
    # particle_images_path_list = get_image_paths(particle_folder)

    frame_width = config.frame_width
    frame_height = config.frame_height

    mod_y = 0
    face_padding = 16

    for frame_index in range(0, len(bg_images_path_list)):
        frame_number = frame_index + 1
        print(f"# frame {frame_number}")

        bg_img = None
        particle_img = None
        output_filename = os.path.basename(bg_images_path_list[frame_index])

        patch_coords = []
        masks = []
        for patch_index, patch_rect in enumerate(patchconfig.patch_rects):
            # print(patch_rect)
            if len(patch_rect) == 4:
                [mode, mode_index, start_frame, end_frame] = patch_rect
                x, y, w, h = 0, 0, 0, 0
            elif len(patch_rect) == 6:
                [x, y, w, h, start_frame, end_frame] = patch_rect
            else:
                break

            if frame_number >= start_frame and frame_number <= end_frame:
                if mode == "face":
                    print(f"face patch")
                    face_img = Image.open(bg_images_path_list[frame_index])
                    face_detect_coords = face_detect(face_img, config.face_threshold)
                    face_coords = []
                    for coord in face_detect_coords:
                        (x1, y1, x2, y2) = coord
                        x1 = int(x1 - face_padding)
                        y1 = int(y1 - face_padding)
                        x2 = int(x2 + face_padding)
                        y2 = int(y2 + face_padding)
                        if x1 < 0:
                            x1 = 0
                        if y1 < 0:
                            y1 = 0
                        if x2 > face_img.width:
                            x2 = face_img.width
                        if y2 > face_img.height:
                            y2 = face_img.height
                        face_coords.append([x1, y1, x2 - x1, y2 - y1])

                    if len(face_coords) > 0:
                        if isinstance(mode_index, str):
                            if mode_index == "left":
                                select_face_coord = None
                                left = face_img.width
                                for i in range(0, len(face_coords)):
                                    (x1, y1, x2, y2) = face_coords[i]
                                    if left > x1:
                                        select_face_coord = face_coords[i]
                                        left = x1
                            if select_face_coord != None:
                                [x, y, w, h] = select_face_coord

                        elif isinstance(mode_index, int):
                            if mode_index <= len(face_coords) - 1:
                                [x, y, w, h] = face_coords[mode_index]

                if w == 0 or h == 0:
                    continue
                if bg_img == None:
                    bg_img = Image.open(bg_images_path_list[frame_index])
                    particle_img = Image.open(os.path.join(particle_folder, os.path.basename(bg_images_path_list[frame_index])))

                    bg_img_arr = np.array(bg_img)
                    particle_img_arr = np.array(particle_img)

                    if bg_img.width != frame_width or bg_img.height != frame_height:
                        bg_img_arr = resize_image(bg_img_arr, frame_width, frame_height, config.frame_resize, config.frame_resize_anchor)
                        bg_img = Image.fromarray(bg_img_arr)

                    if particle_img.width != frame_width or particle_img.height != frame_height:
                        particle_img_arr = resize_image(bg_img_arr, frame_width, frame_height, config.frame_resize, config.frame_resize_anchor)
                        particle_img = Image.fromarray(particle_img_arr)

                    # if bg_img.width != frame_width or bg_img.height != frame_height:
                    #     fit_height = (int)(bg_img.height * (frame_width / bg_img.width))
                    #     if fit_height != bg_img.height:
                    #         bg_img_arr = resize_image(bg_img_arr, frame_width, fit_height)
                    #     if frame_height > fit_height:
                    #         top = (int)((frame_height - fit_height) / 2)
                    #         bottom = (frame_height - fit_height) - top
                    #         bg_img_arr = np.pad(bg_img_arr, pad_width=((top, bottom), (0, 0), (0, 0)), mode="constant")
                    #     elif fit_height > frame_height:
                    #         top = (int)((fit_height - frame_height) / 2)
                    #         bottom = (fit_height - frame_height) - top
                    #         bg_img_arr = bg_img_arr[top:frame_height, 0:frame_width]
                    #     output_bg_filename = f"{os.path.splitext(output_filename)[0]}-cover.png"
                    #     Image.fromarray(bg_img_arr).save(os.path.join(patch_mask_output_folder, output_bg_filename))

                    # if particle_img.width != frame_width or particle_img.height != frame_height:
                    #     particle_img_arr = resize_image(particle_img_arr, frame_width, frame_height)

                mask = Image.new("L", [frame_width, frame_height], 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([x, y + mod_y, x + w, y + h], fill=255)
                masks.append(np.array(mask))
                patch_coords.append([x, y + mod_y, w, h])

        mask_arrs = blur_masks(masks, 20)
        for mask_index, mask_arr in enumerate(mask_arrs):
            mask = Image.fromarray(mask_arr)
            output_mask_filename = f"{os.path.splitext(output_filename)[0]}-{mask_index}.png"
            output_mask_image_path = os.path.join(patch_mask_output_folder, output_mask_filename)
            mask.save(output_mask_image_path)

        for patch_index, (patch_coord, mask_arr) in enumerate(zip(patch_coords, mask_arrs)):
            [x, y, w, h] = patch_coord
            particle_img_arr = particle_img_arr[y : y + h, x : x + w]
            Image.fromarray(particle_img_arr).save(os.path.join(patch_mask_output_folder, output_filename))
            particle_img_arr = merge_image(bg_img_arr, particle_img_arr, patch_coord, mask_arr)

        if len(patch_coords) >= 1:
            merged_image = Image.fromarray(particle_img_arr)
            merged_image.save(os.path.join(patch_output_folder, output_filename))

            print(f"patch image for frame {frame_number}:")
