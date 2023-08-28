import os
import helper.webuiapi as webuiapi
from PIL import Image
import numpy as np
from helper.temporalnet2 import make_flow, encode_image
from helper.zoom import process as zoom_process
from helper.facedetect import face_detect, process as face_process
from helper.image_util import zoom_image, resize_image, crop_and_resize, merge_image
from helper.config import Config
from helper.util import get_image_paths, get_lora_paths, smooth_data, intersect_rect
from shutil import copyfile
import random


class TweenValue:
    def __init__(self, current=0) -> None:
        self.current = current
        self.start = current
        self.goal = current
        self.count = 0
        self.total = 0

    def next(self):
        if self.count >= self.total:
            self.current = self.goal
        else:
            t = self.count / self.total
            self.current = (1 - t) * self.start + t * self.goal
            self.count = self.count + 1

    def reset(self, goal, total, start=None):
        self.goal = goal
        self.total = total
        self.count = 0
        if total == 0 or total == 1:
            self.current = goal
        if start != None:
            self.start = start
        else:
            self.start = self.current

    def isend(self):
        return self.total <= self.count


schedule_availables = [
    "frame_crop",
    "frame_zoom",
    "dynamic_face_zoom",
]


def run(config: Config, project_folder: str):
    print(f"# dynamic_zoom process {project_folder}")

    if project_folder:
        input_folder = os.path.normpath(os.path.join(project_folder, config.input_folder))
        input_zoom_folder = os.path.normpath(os.path.join(project_folder, config.input_folder + "_zoom"))
    else:
        print("project path not found")
        return

    print(f"# input images path {input_folder}")
    print(f"# input zoom folder {input_zoom_folder}")

    os.makedirs(input_zoom_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"# not found input_video_path")

    input_images_path_list = get_image_paths(input_folder)
    total_frames = len(input_images_path_list)
    start_index = 0

    input_img = None
    input_img_arr = None

    tweenScale = TweenValue(1)
    tweenOffsetX = TweenValue(0)
    tweenOffsetY = TweenValue(0)

    # dynamic face zoom
    dynamic_face_zoom_scales = []
    if config.dynamic_face_zoom:
        print(f"# dynamic face zoom mode")
        print(f"  frame zoom skip")
        if os.path.isfile(os.path.join(project_folder, f"./dynamic_face_zoom_scales.npy")):
            dynamic_face_zoom_scales = np.load(os.path.join(project_folder, f"./dynamic_face_zoom_scales.npy")).tolist()

        if len(dynamic_face_zoom_scales) != total_frames:
            dynamic_face_zoom_scales = []
            for frame_index in range(start_index, total_frames):
                frame_width = config.frame_width
                frame_height = config.frame_height

                input_img = Image.open(input_images_path_list[frame_index])
                input_img_arr = np.array(input_img)

                face_detect_coords = face_detect(Image.fromarray(input_img_arr), config.face_threshold)

                select_face_coords = None
                zoom_scale = None
                if len(face_detect_coords) == 1:
                    select_face_coords = face_detect_coords[0]
                    (x1, y1, x2, y2) = select_face_coords
                    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                    select_area = w * h
                elif len(face_detect_coords) > 1:
                    select_face_coords = face_detect_coords[0]
                    (x1, y1, x2, y2) = select_face_coords
                    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                    select_area = w * h
                    for i in range(1, len(face_detect_coords)):
                        (x1, y1, x2, y2) = face_detect_coords[i]
                        (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                        area = w * h
                        print(f"- {area}")
                        if select_area > area:
                            select_face_coords = face_detect_coords[i]
                            select_area = area

                if select_face_coords != None:
                    (x1, y1, x2, y2) = select_face_coords
                    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                    print(f"face detect ({x}, {y}, {w}, {h})")
                    dynamic_face_size = select_area**0.5
                    print(f"dynamic_face_scale {dynamic_face_size}")

                    guide_area_size = frame_width / config.dynamic_face_zoom_ratio
                    print(f"dynamic guide_area_scale {guide_area_size}")
                    if guide_area_size > dynamic_face_size:
                        zoom_scale = 1 + (guide_area_size - dynamic_face_size) / guide_area_size
                        print(f"dynamic zoom_scale {zoom_scale}")
                    elif guide_area_size < dynamic_face_size:
                        if tweenScale.current > 1:
                            zoom_scale = 1 + (guide_area_size - dynamic_face_size) / guide_area_size
                            print(f"dynamic zoom_scale {zoom_scale}")

                if zoom_scale == None:
                    if frame_index == 0:
                        zoom_scale = 1
                    else:
                        zoom_scale = dynamic_face_zoom_scales[frame_index - 1]
                    if zoom_scale < 1:
                        zoom_scale = 1

                dynamic_face_zoom_scales.append(zoom_scale)

            print(len(dynamic_face_zoom_scales))
            dynamic_face_zoom_scales = smooth_data(dynamic_face_zoom_scales, 5)
            print(len(dynamic_face_zoom_scales))
            np.save(os.path.join(project_folder, f"./dynamic_face_zoom_scales.npy"), dynamic_face_zoom_scales)

    for frame_index in range(start_index, total_frames):
        output_filename = os.path.basename(input_images_path_list[frame_index])
        output_image_path = os.path.join(input_zoom_folder, output_filename)

        frame_number = frame_index + 1
        print(f"# frame {frame_number}/{total_frames}")

        #####################
        # frame schedule
        if str(frame_number) in config.frame_schedule:
            print(f"# frame schedule {frame_number}")
            frame_config = config.frame_schedule[str(frame_number)]
            if "rollback" in frame_config:
                print(f"# rollback")
                frame_config = {}
                for key in schedule_availables:
                    frame_config[key] = getattr(config, key)

            if "break" in frame_config:
                break

            for key in schedule_availables:
                if key in frame_config:
                    setattr(config, key, frame_config[key])

        frame_width = config.frame_width
        frame_height = config.frame_height

        input_img = Image.open(input_images_path_list[frame_index])
        input_img_arr = np.array(input_img)

        # image crop
        if config.frame_crop != None and len(config.frame_crop) == 4:
            [x, y, w, h] = config.frame_crop
            input_img_arr = input_img_arr[y : y + h, x : x + w]
            input_img = Image.fromarray(input_img_arr)
            # input_img.save(os.path.join(output_folder, output_filename))

        # fit frame size
        if input_img.width != frame_width or input_img.height != frame_height:
            input_img_arr = resize_image(input_img_arr, frame_width, frame_height, config.frame_resize, config.frame_resize_anchor)
            input_img = Image.fromarray(input_img_arr)

        # dynamic face zoom
        if config.dynamic_face_zoom:
            scale = dynamic_face_zoom_scales[frame_index] if len(dynamic_face_zoom_scales) > frame_index else dynamic_face_zoom_scales[-1]
            print(f"scale {scale}")
            input_img_arr = zoom_image(input_img_arr, scale)
            (h, w) = input_img_arr.shape[:2]
            if config.dynamic_face_zoom_anchor == "left":
                x = 0
            else:
                x = (w - input_img.width) >> 1
            y = (h - input_img.height) >> 1
            input_img_arr = input_img_arr[y : y + frame_height, x : x + frame_width]
            input_img = Image.fromarray(input_img_arr)

        # zoom scale
        else:
            if config.frame_zoom != None:
                print(f"frame zoom {config.frame_zoom}")
                if isinstance(config.frame_zoom, list):
                    if len(config.frame_zoom) == 4:
                        [scale, offset_x, offset_y, frames] = config.frame_zoom
                    elif len(config.frame_zoom) == 3:
                        [scale, offset_x, offset_y] = config.frame_zoom
                        frames = 0
                    elif len(config.frame_zoom) == 1:
                        [scale] = config.frame_zoom
                        offset_x = 0
                        offset_y = 0
                        frames = 0
                    else:
                        scale = 1
                        offset_x = 0
                        offset_y = 0
                        frames = 0
                else:
                    scale = config.frame_zoom
                    offset_x = 0
                    offset_y = 0
                    frames = 0
                if scale < 1:
                    scale = 1

                tweenScale.reset(scale, frames)
                tweenOffsetX.reset(offset_x, frames)
                tweenOffsetX.reset(offset_y, frames)
                tweenScale.next()
                tweenOffsetX.next()
                tweenOffsetY.next()
                config.frame_zoom = None

            print(f"scale {tweenScale.current}")
            if tweenScale.current != 1 or tweenOffsetX.current != 0 or tweenOffsetY.current != 0:
                print(f"zoom scale {tweenScale.current} ({tweenOffsetX.current},{tweenOffsetY.current})")
                input_img_arr = zoom_image(input_img_arr, tweenScale.current)
                (h, w) = input_img_arr.shape[:2]
                x = ((w - input_img.width) >> 1) + int(tweenOffsetX.current)
                y = ((h - input_img.height) >> 1) + int(tweenOffsetY.current)
                input_img_arr = input_img_arr[y : y + frame_height, x : x + frame_width]
                input_img = Image.fromarray(input_img_arr)

            tweenScale.next()
            tweenOffsetX.next()
            tweenOffsetY.next()

        input_img.save(output_image_path)
