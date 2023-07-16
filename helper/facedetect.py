import os
from PIL import Image, ImageDraw
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from helper.image_util import blur_masks, resize_image

model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
face_model = YOLO(model_path)


def face_detect(image, conf_thres):
    output = face_model(image, conf=conf_thres)
    bboxes = output[0].boxes.xyxy.cpu().numpy()
    if bboxes.size > 0:
        bboxes = bboxes.tolist()
    return bboxes


def x_ceiling(value, step):
    return -(-value // step) * step


def face_img_crop(img_array, face_coords):
    face_crop_resolution = 512
    face_imgs = []
    new_coords = []

    for face in face_coords:
        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])
        # print(f"face {[x,y,w,h]}")

        if w * h <= 150000:
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
        face_img = resize_image(face_img, re_w, re_h)
        resized.append(Image.fromarray(face_img))
        new_coord[2] = calc_w
        new_coord[3] = calc_h

    # print(new_coords)
    return resized, new_coords


def process(input_img_arr, threshold, padding, face_image_folder, output_filename, coords=None):
    (height, width) = input_img_arr.shape[:2]
    output_basename = os.path.splitext(output_filename)[0]
    if coords == None:
        coords = face_detect(Image.fromarray(input_img_arr), threshold)
    face_coords = []
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
        if x2 > width:
            x2 = width
        if y2 > height:
            y2 = height
        (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
        face_coords.append((x, y, w, h))
    # print(face_coords)
    face_ret = face_img_crop(input_img_arr, face_coords)
    for face_index, face in enumerate(face_ret[0]):
        output_face_filename = f"{output_basename}-face{face_index}.png"
        output_face_image_path = os.path.join(face_image_folder, output_face_filename)
        face.save(output_face_image_path)

    masks = []
    for mask_index, new_coord in enumerate(face_ret[1]):
        mask = Image.new("L", [height, height], 0)
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

    masks = blur_masks(masks, 25)
    for mask_index, mask in enumerate(masks):
        output_mask_filename = f"{output_basename}-mask{mask_index}.png"
        output_mask_image_path = os.path.join(face_image_folder, output_mask_filename)
        mask.save(output_mask_image_path)

    return face_ret[0], face_ret[1], masks
