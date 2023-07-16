import os
import helper.webuiapi as webuiapi
from PIL import Image
import numpy as np
from helper.temporalnet2 import make_flow
from helper.temporalnet2 import encode_image
from helper.zoom import process as zoom_process
from helper.facedetect import face_detect, process as face_process
from helper.image_util import zoom_image, resize_image, crop_and_resize, merge_image
from helper.config import Config
from helper.util import get_image_paths
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
            self.current = this.goal
        else:
            t = self.count / self.total
            self.current = (1 - t) * self.start + t * self.goal
            self.count = self.count + 1

    def reset(self, goal, total, start=None):
        self.goal = goal
        self.total = total
        self.count = 0
        if start != None:
            self.start = start
        else:
            self.start = current

    def isend(self):
        return self.total <= self.count


schedule_availables = [
    "base_prompt",
    "face_prompt",
    "seed",
    "seed_mode",
    "sampler_name",
    "sampler_step",
    "face_sampler_step",
    "cfg_scale",
    "frame_crop",
    "frame_zoom",
    "use_base_img2img",
    "use_face_img2img",
    "use_zoom_img2img",
    "denoising_strength",
    "face_denoising_strength",
    "face_threshold",
    "face_padding",
    "zoom_denoising_strength",
    "use_interrogate",
    "use_face_interrogate",
    "temporalnet",
    "temporalnet_weight",
    "zoom_temporalnet",
    "zoom_temporalnet_weight",
    "face_temporalnet",
    "face_temporalnet_weight",
]


unit_tempo_v1 = webuiapi.ControlNetUnit(
    module="none",
    model="diff_control_sd15_temporalnet_fp16 [adc6bd97]",
    weight=1,
    control_mode=0,
    pixel_perfect=True,
)
unit_tempo_v2 = webuiapi.ControlNetUnit(
    module="none",
    model="temporalnetversion2 [b146ac48]",
    weight=1,
    threshold_a=64,
    threshold_b=64,
    control_mode=0,
    pixel_perfect=True,
)

api = webuiapi.WebUIApi()


def run(config: Config, project_folder: str, overwrite: bool, resume_frame: int, end_frame: int, rework_mode: str = None):
    print(f"# project path {project_folder}")

    if project_folder:
        input_folder = os.path.normpath(os.path.join(project_folder, config.input_folder))
        output_folder = os.path.normpath(os.path.join(project_folder, config.output_folder))
        zoom_image_folder = os.path.normpath(os.path.join(project_folder, config.zoom_image_folder))
        face_image_folder = os.path.normpath(os.path.join(project_folder, config.face_image_folder))
        flow_image_folder = os.path.normpath(os.path.join(project_folder, config.flow_image_folder))
    else:
        input_folder = config.input_folder
        output_folder = config.output_folder
        zoom_image_folder = config.zoom_image_folder
        face_image_folder = config.face_image_folder
        flow_image_folder = config.flow_image_folder

    print(f"# input images path {input_folder}")
    print(f"# output folder {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(zoom_image_folder, exist_ok=True)
    os.makedirs(face_image_folder, exist_ok=True)
    os.makedirs(flow_image_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"# extract input_video_path")

    if config.seed == -1:
        seed = random.randrange(1, 2**63)
    else:
        seed = config.seed
    print(f"# seed {seed}")
    print(f"# seed mode {config.seed_mode}")

    if config.checkpoint:
        api.refresh_checkpoints()
        print(f"# change checkpoint {config.checkpoint}")
        api.util_set_model(config.checkpoint)

    current_model = api.util_get_current_model()
    print(f"# current checkpoint {current_model}")

    controlnet_units = []
    for cn in config.controlnet:
        cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=config.controlnet_lowvram)
        controlnet_units.append(cn_unit)

    zoom_controlnet_units = []
    for cn in config.zoom_controlnet:
        cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=config.controlnet_lowvram)
        zoom_controlnet_units.append(cn_unit)

    face_controlnet_units = []
    for cn in config.face_controlnet:
        cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=config.controlnet_lowvram)
        face_controlnet_units.append(cn_unit)

    init_image = None
    if config.init_image_path:
        init_image = Image.open(os.path.join(project_folder, config.init_image_path))

    input_images_path_list = get_image_paths(input_folder)

    input_img = None
    input_img_arr = None

    last_image_arr = None
    flow_image_arr = None

    base_output_image = None
    base_output_image_arr = None

    start_index = 0
    if init_image != None:
        last_image_arr = np.array(init_image)
        start_index = 1

    tweenScale = TweenValue(1)
    tweenOffsetX = TweenValue(0)
    tweenOffsetY = TweenValue(0)

    if not config.start_frame:
        config.start_frame = 1

    total_frames = len(input_images_path_list)
    for i in range(start_index, total_frames):
        frame_number = i + 1
        print(f"# frame {frame_number}")

        if config.start_frame > frame_number:
            continue

        output_filename = os.path.basename(input_images_path_list[i])
        output_image_path = os.path.join(output_folder, output_filename)

        # init
        face_detect_coords = None

        #####################
        # frame schedule
        if str(frame_number) in config.frame_schedule:
            print(f"# frame schedule {frame_number}")
            frame_config = config.frame_schedule[str(frame_number)]
            if "checkpoint" in frame_config:
                checkpoint = frame_config["checkpoint"]
                print(f"# change checkpoint {checkpoint}")
                api.util_set_model(checkpoint)

            if "seed" in frame_config:
                if frame_config["seed"] == -1:
                    seed = random.randrange(1, 2**63)
                else:
                    seed = frame_config["seed"]
                print(f"# seed {seed}")

            if "rollback" in frame_config:
                frame_config = config
                checkpoint = frame_config["checkpoint"]
                print(f"# change checkpoint {checkpoint}")
                api.util_set_model(checkpoint)
                print(f"# rollback")

            if "break" in frame_config:
                break

            for key in schedule_availables:
                if key in frame_config:
                    setattr(config, key, frame_config[key])

        frame_width = config.frame_width
        frame_height = config.frame_height

        input_img = Image.open(input_images_path_list[i])
        input_img_arr = np.array(input_img)

        # image crop
        if config.frame_crop != None and len(config.frame_crop) == 4:
            [x, y, w, h] = config.frame_crop
            input_img_arr = input_img_arr[y : y + h, x : x + w]
            input_img = Image.fromarray(input_img_arr)
            # input_img.save(os.path.join(output_folder, output_filename))

        # fit frame size
        if input_img.width != config.frame_width or input_img.height != config.frame_height:
            input_img_arr = resize_image(input_img_arr, config.frame_width, config.frame_height, config.frame_resize, config.frame_resize_anchor)
            input_img = Image.fromarray(input_img_arr)

        # dynamic face zoom
        if config.dynamic_face_zoom:
            face_detect_coords = face_detect(Image.fromarray(input_img_arr), config.face_threshold)

            select_face_coords = None
            if len(face_detect_coords) == 1:
                select_face_coords = face_detect_coords[0]
            elif len(face_detect_coords) > 1:
                select_face_coords = face_detect_coords[0]
                select_area = select_face_coords[2] * select_face_coords[3]
                for i in range(1, len(face_detect_coords)):
                    (x1, y1, x2, y2) = face_detect_coords[i]
                    (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                    print(f"{w}x{h}")
                    area = w * h
                    if select_area > area:
                        select_face_coords = face_detect_coords[i]
                        select_area = area

            if select_face_coords != None:
                (x1, y1, x2, y2) = select_face_coords
                (x, y, w, h) = (x1, y1, x2 - x1, y2 - y1)
                # print(f"face detect ({x}, {y}, {w}, {h})")
                dynamic_face_size = select_area**0.5
                # print(f"dynamic_face_scale {dynamic_face_size}")

                guide_area_size = frame_width / 7
                # print(f"dynamic guide_area_scale {guide_area_size}")
                if guide_area_size > dynamic_face_size:
                    zoom_scale = 1 + (guide_area_size - dynamic_face_size) / guide_area_size
                    # print(f"dynamic zoom_scale {zoom_scale}")
                    config.frame_zoom = [zoom_scale, 0, 0, 10]
                elif guide_area_size < dynamic_face_size:
                    if tweenScale.current > 1:
                        zoom_scale = 1 + (guide_area_size - dynamic_face_size) / guide_area_size
                        # print(f"dynamic zoom_scale {zoom_scale}")
                        config.frame_zoom = [zoom_scale, 0, 0, 10]

        # zoom scale
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
            config.frame_zoom = None

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

        # resume frame
        if frame_number < resume_frame:
            if frame_number == resume_frame - 1:
                last_image_arr = np.array(Image.open(output_image_path))
            continue

        if not overwrite:
            if os.path.isfile(output_image_path):
                print("skip")
                if frame_number < total_frames and not os.path.isfile(os.path.join(output_folder, os.path.basename(input_images_path_list[i + 1]))):
                    last_image_arr = Image.open(os.path.join(output_folder, os.path.basename(input_images_path_list[i])))
                continue

        ########################
        # base img2img
        if config.use_interrogate:
            ret = api.interrogate(input_img, config.interrogate_model)
            print(f"[interrogate({config.interrogate_model})] {ret.info}")
            prompt = config.base_prompt + ret.info
        else:
            prompt = config.base_prompt

        if config.use_base_img2img:
            for unit in controlnet_units:
                unit.input_image = input_img

            p_controlnet_units = []
            if config.start_frame == frame_number or (config.temporalnet_reset_frames != None and frame_number in config.temporalnet_reset_frames):
                if config.temporalnet_reset_interrogate:
                    ret = api.interrogate(input_img, config.interrogate_model)
                    print(f"[temporalnet reset interrogate({config.interrogate_model})] {ret.info}")
                    prompt = config.base_prompt + ret.info

                p_controlnet_units = controlnet_units
            else:
                unit_tempo = None
                if config.temporalnet == "v2":
                    flow_image_arr = make_flow(input_images_path_list[i - 1], input_images_path_list[i], frame_width, frame_height, flow_image_folder, output_filename)
                    unit_tempo = unit_tempo_v2
                    unit_tempo.weight = config.temporalnet_weight
                    unit_tempo.encoded_image = encode_image(flow_image_arr, last_image_arr)
                    unit_tempo.lowvram = config.controlnet_lowvram
                elif config.temporalnet == "v1":
                    unit_tempo = unit_tempo_v1
                    unit_tempo.weight = config.temporalnet_weight
                    unit_tempo.input_image = Image.fromarray(last_image_arr)
                    unit_tempo.lowvram = config.controlnet_lowvram
                p_controlnet_units = controlnet_units + [unit_tempo]

            ret = api.img2img(
                prompt=prompt,
                negative_prompt=config.neg_prompt,
                sampler_name=config.sampler_name,
                steps=config.sampler_step,
                images=[input_img],
                denoising_strength=config.denoising_strength,
                seed=-1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + i,
                cfg_scale=config.cfg_scale,
                width=frame_width,
                height=frame_height,
                controlnet_units=[x for x in p_controlnet_units if x is not None],
            )

            base_output_image = ret.images[0]
            base_output_image_arr = np.array(base_output_image)
            base_output_image.save(output_image_path)
        else:
            if rework_mode:
                if rework_mode == "zoom":
                    if os.path.exists(os.path.join(zoom_image_folder, output_filename)):
                        base_output_image = Image.open(os.path.join(zoom_image_folder, output_filename))
                        base_output_image_arr = np.array(base_output_image)
                    elif os.path.exists(os.path.join(face_image_folder, output_filename)):
                        base_output_image = Image.open(os.path.join(face_image_folder, output_filename))
                        base_output_image_arr = np.array(base_output_image)
                    elif os.path.exists(output_image_path):
                        base_output_image = Image.open(output_image_path)
                        base_output_image_arr = np.array(base_output_image)
                elif rework_mode == "face":
                    if os.path.exists(os.path.join(face_image_folder, output_filename)):
                        base_output_image = Image.open(os.path.join(face_image_folder, output_filename))
                        base_output_image_arr = np.array(base_output_image)
                    elif os.path.exists(output_image_path):
                        base_output_image = Image.open(output_image_path)
                        base_output_image_arr = np.array(base_output_image)
                elif rework_mode == "patch":
                    if os.path.exists(output_image_path):
                        base_output_image = Image.open(output_image_path)
                        base_output_image_arr = np.array(base_output_image)

            else:
                base_output_image = input_img
                base_output_image_arr = np.array(base_output_image)
                base_output_image.save(output_image_path)

        ######################
        # zoom img2img
        if config.use_zoom_img2img:
            (zoom_images, zoom_coords, zoom_masks) = zoom_process(i, input_img_arr, config.zoom_rects, config.zoom_area_limit, config.zoom_max_resolusion, zoom_image_folder, output_filename)

            for zoom_index, (zoom_img, zoom_coord, mask) in enumerate(zip(zoom_images, zoom_coords, zoom_masks)):
                [x, y, re_w, re_h, calc_w, calc_h] = zoom_coord

                for unit in zoom_controlnet_units:
                    unit.input_image = zoom_img

                p_zoom_controlnet_units = []
                if config.start_frame == frame_number or (config.temporalnet_reset_frames != None and frame_number in config.temporalnet_reset_frames):
                    p_zoom_controlnet_units = zoom_controlnet_units
                else:
                    unit_tempo = None
                    last_zoom_img_arr = crop_and_resize(last_image_arr, [x, y, calc_w, calc_h], zoom_img.width, zoom_img.height)
                    if config.zoom_temporalnet == "v2":
                        flow_zoom_image_arr = crop_and_resize(flow_image_arr, [x, y, calc_w, calc_h], zoom_img.width, zoom_img.height)
                        unit_tempo = unit_tempo_v2
                        unit_tempo.weight = config.zoom_temporalnet_weight
                        unit_tempo.encoded_image = encode_image(flow_zoom_image_arr, last_zoom_img_arr)
                        unit_tempo.lowvram = config.controlnet_lowvram

                    elif config.zoom_temporalnet == "v1":
                        unit_tempo = unit_tempo_v1
                        unit_tempo.weight = config.zoom_temporalnet_weight
                        unit_tempo.input_image = Image.fromarray(last_zoom_img_arr)
                        unit_tempo.lowvram = config.controlnet_lowvram
                    p_zoom_controlnet_units = zoom_controlnet_units + [unit_tempo]
                print(f" * zoom {zoom_index}")

                ret = api.img2img(
                    prompt=prompt,
                    negative_prompt=config.neg_prompt,
                    sampler_name=config.sampler_name,
                    steps=config.sampler_step,
                    images=[zoom_img],
                    denoising_strength=config.zoom_denoising_strength,
                    seed=-1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + i,
                    cfg_scale=config.cfg_scale,
                    width=re_w,
                    height=re_h,
                    controlnet_units=[x for x in p_zoom_controlnet_units if x is not None],
                )

                output_zoom_filename = f"{os.path.splitext(output_filename)[0]}-zoom{zoom_index}.png"
                output_zoom_image_path = os.path.join(zoom_image_folder, output_zoom_filename)
                zoom_img.save(output_zoom_image_path)
                output_zoom_filename = f"{os.path.splitext(output_filename)[0]}-zoom-convert{zoom_index}.png"
                output_zoom_image_path = os.path.join(zoom_image_folder, output_zoom_filename)
                zoom_output_image = ret.images[0]
                zoom_output_image.save(output_zoom_image_path)
                zoom_output_image_arr = np.array(zoom_output_image)

                base_output_image_arr = merge_image(base_output_image_arr, zoom_output_image_arr, (x, y, calc_w, calc_h), mask)

            if len(zoom_images) > 0:
                output_full_image_path = os.path.join(zoom_image_folder, output_filename)
                copyfile(output_image_path, output_full_image_path)
                base_output_image = Image.fromarray(base_output_image_arr)
                base_output_image.save(output_image_path)

        #####################
        # face img2img
        if config.use_face_img2img:
            (face_imgs, face_coords, face_masks) = face_process(input_img_arr, config.face_threshold, config.face_padding, face_image_folder, output_filename, face_detect_coords)

            for face_index, (face_img, face_coord, mask) in enumerate(zip(face_imgs, face_coords, face_masks)):
                for unit in face_controlnet_units:
                    unit.input_image = face_img

                p_face_controlnet_units = []
                if config.start_frame == frame_number or (config.temporalnet_reset_frames != None and frame_number in config.temporalnet_reset_frames):
                    p_face_controlnet_units = face_controlnet_units
                else:
                    unit_tempo = None
                    last_face_img_arr = crop_and_resize(last_image_arr, face_coord, face_img.width, face_img.height)
                    if config.face_temporalnet == "v2":
                        flow_face_image_arr = crop_and_resize(flow_image_arr, face_coord, face_img.width, face_img.height)
                        unit_tempo = unit_tempo_v2
                        unit_tempo.weight = config.face_temporalnet_weight
                        unit_tempo.lowvram = config.controlnet_lowvram
                        unit_tempo.encoded_image = encode_image(flow_face_image_arr, last_face_img_arr)
                    elif config.face_temporalnet == "v1":
                        unit_tempo = unit_tempo_v1
                        unit_tempo.weight = config.face_temporalnet_weight
                        unit_tempo.input_image = Image.fromarray(last_face_img_arr)
                        unit_tempo.lowvram = config.controlnet_lowvram
                    p_face_controlnet_units = face_controlnet_units + [unit_tempo]

                if config.use_face_interrogate:
                    ret = api.interrogate(face_img, config.face_interrogate_model)
                    print(f"[face-interrogate({config.face_interrogate_model})] {ret.info}")
                    prompt = config.face_prompt + ret.info
                else:
                    prompt = config.face_prompt
                print(f" * face {face_index}")

                ret = api.img2img(
                    prompt=prompt,
                    negative_prompt=config.neg_prompt,
                    sampler_name=config.sampler_name,
                    steps=config.face_sampler_step,
                    images=[face_img],
                    denoising_strength=config.face_denoising_strength,
                    seed=-1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + i,
                    cfg_scale=config.cfg_scale,
                    width=face_img.width,
                    height=face_img.height,
                    controlnet_units=[x for x in p_face_controlnet_units if x is not None],
                )

                output_face_filename = f"{os.path.splitext(output_filename)[0]}-face-convert{face_index}.png"
                output_face_image_path = os.path.join(face_image_folder, output_face_filename)
                face_output_image = ret.images[0]
                face_output_image.save(output_face_image_path)
                face_output_image_arr = np.array(face_output_image)

                base_output_image_arr = merge_image(base_output_image_arr, face_output_image_arr, new_coord, mask)

            if len(face_imgs) > 0:
                output_full_image_path = os.path.join(face_image_folder, output_filename)
                copyfile(output_image_path, output_full_image_path)
                base_output_image = Image.fromarray(base_output_image_arr)
                base_output_image.save(output_image_path)

        base_output_image = base_output_image_arr
        last_image_arr = base_output_image_arr

        if end_frame > 0 and end_frame == frame_number:
            break
