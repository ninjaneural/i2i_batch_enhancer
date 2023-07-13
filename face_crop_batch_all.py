import os
import glob
import helper.webuiapi as webuiapi
from PIL import Image, ImageDraw
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
face_model = YOLO(model_path)

face_prompt = "(masterpiece,best quality:1.2), face close up,"
neg_prompt = "(worst quality, low quality:1.2),"

seed = -1
# sampler_name = "DPM++ SDE Karras"
sampler_name = "Euler a"
face_sampler_step = 30
cfg_scale = 5
face_denoising_strength = 0.65
face_threshold = 0.25
face_padding = 32
exists_skip = True

use_face_crop = True
use_face_interrogate = True

unit_tile = webuiapi.ControlNetUnit(
    module="tile_resample",
    model="control_v11f1e_sd15_tile_fp16 [3b860298]",
    weight=1,
    down_sampling_rate=1,
    control_mode=0,
)
unit_lineart = webuiapi.ControlNetUnit(
    module="lineart_realistic",
    model="control_v11p_sd15_lineart_fp16 [5c23b17d]",
    weight=0.8,
    control_mode=2,
)
unit_openpose = webuiapi.ControlNetUnit(
    module="openpose",
    model="control_v11p_sd15_openpose_fp16 [73c2b67d]",
    weight=0.7,
    lowvram=True,
)
unit_depth = webuiapi.ControlNetUnit(
    # module="depth_leres++",
    module="none",
    model="control_v11p_sd15_depth_fp16 [12c052a1]",
    weight=1,
    control_mode=2,
    lowvram=True,
)
unit_softedge = webuiapi.ControlNetUnit(
    module="softedge_hed",
    model="control_v11p_sd15_softedge_fp16 [f616a34f]",
    weight=0.7,
    control_mode=2,
    lowvram=True,
)
unit_inpaint = webuiapi.ControlNetUnit(
    module="inpaint_only",
    model="control_v11p_sd15_inpaint_fp16 [be8bc0ed]",
    weight=0.8,
    control_mode=1,
    lowvram=True,
)

input_controlnet_face_units = [unit_tile, unit_lineart]

api = webuiapi.WebUIApi()


def convert(input_images_path="./input", output_folder=""):
    if input_images_path[-7:] == "_output":
        return

    if output_folder == "":
        output_folder = input_images_path + "_output"

    os.makedirs(output_folder, exist_ok=True)

    def face_detect(image, conf_thres):
        output = face_model(image, conf=conf_thres)
        bboxes = output[0].boxes.xyxy.cpu().numpy()
        conf = output[0].boxes.conf.cpu().numpy()
        if bboxes.size > 0:
            bboxes = bboxes.tolist()
        return [conf, bboxes]

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

    def x_ceiling(value, step):
        return -(-value // step) * step

    def resize_img(img, w, h):
        if img.shape[0] + img.shape[1] < h + w:
            interpolation = interpolation = cv2.INTER_CUBIC
        else:
            interpolation = interpolation = cv2.INTER_AREA

        return cv2.resize(img, (w, h), interpolation=interpolation)

    def face_img_crop(img, face_coords):
        face_crop_resolution = 512
        img_array = np.array(img)
        face_imgs = []
        new_coords = []

        for face in face_coords:
            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])
            print(f"face {[x,y,w,h]}")

            if w * h <= 15000000:
                face_imgs.append(img_array[y : y + h, x : x + w])
                new_coords.append([x, y, w, h])

        resized = []
        for face_img, new_coord in zip(face_imgs, new_coords):
            [x, y, w, h] = new_coord
            re_w = face_crop_resolution
            re_h = int(
                x_ceiling(
                    (face_crop_resolution / w) * h,
                    64,
                )
            )
            calc_w = w
            calc_h = int(re_h / (face_crop_resolution / w))

            # print(f"resize {[re_w,re_h]}")
            # print(f"calc {[calc_w,calc_h]}")
            face_img = img_array[y : y + calc_h, x : x + calc_w]
            face_img = resize_img(face_img, re_w, re_h)
            resized.append(Image.fromarray(face_img))
            new_coord[2] = calc_w
            new_coord[3] = calc_h

        # print(new_coords)
        return resized, new_coords

    def merge_face(img, face_img, face_coord, mask):
        img_array = np.array(img.convert("RGB"))
        (x, y, w, h) = face_coord
        # print(f"merge {w}x{h}")

        face_array = np.array(face_img.convert("RGB"))
        face_array = np.array(resize_img(face_array, w, h))
        mask_array = np.array(mask.convert("L"))
        mask_array = mask_array[y : y + h, x : x + w]

        mask_array = mask_array.astype(dtype="float") / 255
        # mask_array = mask_array[]
        # print(mask_array)
        if mask_array.ndim == 2:
            mask_array = mask_array[:, :, np.newaxis]

        (
            h,
            w,
        ) = mask_array.shape[:2]
        face_array = face_array[:h, :w]

        bg = img_array[y : y + h, x : x + w]
        img_array[y : y + h, x : x + w] = mask_array * face_array + (1 - mask_array) * bg

        return Image.fromarray(img_array)

    def get_image_paths(folder):
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for ext in image_extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(files)

    input_images_path_list = get_image_paths(input_images_path)

    for i in range(0, len(input_images_path_list)):
        print(f"input_image:{input_images_path_list[i]}")
        input_img = Image.open(input_images_path_list[i])

        output_filename = os.path.basename(input_images_path_list[i])
        output_image_path = os.path.join(output_folder, output_filename)

        if os.path.isfile(output_image_path) and exists_skip == True:
            continue

        whole_output_image = input_img
        whole_output_image.save(output_image_path)

        ##############
        # face crop
        if use_face_crop:
            [face_conf, coords] = face_detect(input_img, face_threshold)
            print(f"face_conf {face_conf}")
            face_coords = []
            padding = face_padding
            for coord in coords:
                (x1, y1, x2, y2) = coord
                x1 = x1 - padding
                y1 = y1 - padding
                x2 = x2 + padding
                y2 = y2 + padding
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > input_img.width:
                    x2 = input_img.width
                if y2 > input_img.height:
                    y2 = input_img.height
                (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                face_coords.append((x, y, w, h))
            # print(face_coords)
            face_ret = face_img_crop(input_img, face_coords)
            for face_index, face in enumerate(face_ret[0]):
                output_face_filename = f"{os.path.splitext(output_filename)[0]}-crop{face_index}.png"
                output_face_image_path = os.path.join(output_folder, output_face_filename)
                # face.save(output_face_image_path)

            masks = []
            for mask_index, new_coord in enumerate(face_ret[1]):
                mask = Image.new("L", [input_img.width, input_img.height], 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(
                    [
                        new_coord[0],
                        new_coord[1],
                        new_coord[0] + new_coord[2],
                        new_coord[1] + new_coord[3],
                    ],
                    fill=255,
                )
                masks.append(mask)
                output_mask_filename = f"{os.path.splitext(output_filename)[0]}-msk{mask_index}.png"
                output_mask_image_path = os.path.join(output_folder, output_mask_filename)
                # mask.save(output_mask_image_path)
            masks = blur_masks(masks, 25)
            for mask_index, mask in enumerate(masks):
                output_mask_filename = f"{os.path.splitext(output_filename)[0]}-mask{mask_index}.png"
                output_mask_image_path = os.path.join(output_folder, output_mask_filename)
                # mask.save(output_mask_image_path)

            for face_index, (face_img, new_coord, mask) in enumerate(zip(face_ret[0], face_ret[1], masks)):
                for unit in input_controlnet_face_units:
                    unit.input_image = face_img

                controlnet_face_units = input_controlnet_face_units

                if use_face_interrogate:
                    ret = api.interrogate(face_img, "deepdanbooru")
                    prompt = face_prompt + ret.info
                    print(f"    face {face_index} {ret.info}")
                else:
                    prompt = face_prompt
                    print(f"    face {face_index}")
                ret = api.img2img(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    sampler_name=sampler_name,
                    steps=face_sampler_step,
                    images=[face_img],
                    denoising_strength=face_denoising_strength,
                    seed=seed if seed == -1 else seed + i,
                    cfg_scale=cfg_scale,
                    width=face_img.width,
                    height=face_img.height,
                    controlnet_units=[x for x in controlnet_face_units if x is not None],
                )

                output_face_filename = f"{os.path.splitext(output_filename)[0]}-face{face_index}.png"
                output_face_image_path = os.path.join(output_folder, output_face_filename)
                face_output_image = ret.images[0]
                # face_output_image = Image.open(output_face_image_path)
                # face_output_image.save(output_face_image_path)

                whole_output_image = merge_face(whole_output_image, face_output_image, new_coord, mask)

            if len(face_ret[0]) > 0:
                # output_full_filename = f"{os.path.splitext(output_filename)[0]}-whole.png"
                # output_full_image_path = os.path.join(output_folder, output_full_filename)
                # copyfile(output_image_path, output_full_image_path)
                whole_output_image.save(output_image_path)

        print(f"Written data for frame {i}:")


for path in glob.glob("./*"):
    if os.path.isdir(path):
        convert(path)
