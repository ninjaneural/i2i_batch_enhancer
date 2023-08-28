import os
import helper.webuiapi as webuiapi
from PIL import Image
import numpy as np
from helper.temporalnet2 import make_flow, encode_image
from helper.zoom import process as zoom_process
from helper.facedetect import process as face_process
from helper.image_util import resize_image, crop_and_resize, merge_image
from helper.config import Config
from helper.util import get_image_paths
from shutil import copyfile
import random

schedule_availables = [
    "base_prompt",
    "face_prompt",
    "seed",
    "seed_mode",
    "sampler_name",
    "sampler_step",
    "face_sampler_step",
    "cfg_scale",
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


def run(config: Config, project_folder: str, overwrite: bool, reverse: bool, resume_frame: int, end_frame: int, rework_mode: str = None):
    print(f"# project path {project_folder}")

    if project_folder:
        input_folder = os.path.normpath(os.path.join(project_folder, config.input_folder))
        output_folder = os.path.normpath(os.path.join(project_folder, config.output_folder))
        zoom_image_folder = os.path.normpath(os.path.join(output_folder, "./zoom_images"))
        face_image_folder = os.path.normpath(os.path.join(output_folder, "./face_images"))
        flow_image_folder = os.path.normpath(os.path.join(output_folder, "./flow_images"))
    else:
        print("project path not found")
        return

    print(f"# input images path {input_folder}")
    print(f"# output folder {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(zoom_image_folder, exist_ok=True)
    os.makedirs(face_image_folder, exist_ok=True)
    os.makedirs(flow_image_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"# not found input_video_path")

    if config.seed == -1:
        seed = random.randrange(1, 2**31)
    else:
        seed = config.seed
    if config.subseed == -1:
        subseed = random.randrange(1, 2**31)
    else:
        subseed = config.subseed
    print(f"# seed {seed}")
    print(f"# seed mode {config.seed_mode}")

    # temporalnet v2 bug
    # temporalnet 적용후 controlnet 안쓸때 에러

    if config.checkpoint:
        if config.use_base_img2img or config.use_zoom_img2img or config.use_face_img2img:
            api.refresh_checkpoints()
            print(f"# change checkpoint {config.checkpoint}")
            api.util_set_model(config.checkpoint)

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
    if reverse:
        input_images_path_list.reverse()

    input_img = None
    input_img_arr = None

    last_image_arr = None
    flow_image_arr = None

    base_output_image = None
    base_output_image_arr = None

    last_face_coords = {}

    start_index = 0
    if init_image != None:
        last_image_arr = np.array(init_image)

    if not config.start_frame:
        config.start_frame = 1

    total_frames = len(input_images_path_list)
    print(f"total frames {total_frames}")

    base_prompt = config.base_prompt + "," + config.base_prompt2 + "," + config.base_prompt3 + "," + config.base_prompt4 + "," + config.base_prompt5
    interrogate_prompt = ""

    for frame_index in range(start_index, total_frames):
        output_filename = os.path.basename(input_images_path_list[frame_index])
        output_image_path = os.path.join(output_folder, output_filename)

        frame_number = frame_index + 1
        print(f"# frame {frame_number}/{total_frames}")

        if config.start_frame > frame_number:
            continue

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

            if "checkpoint" in frame_config:
                checkpoint = frame_config["checkpoint"]
                if config.use_base_img2img or config.use_zoom_img2img or config.use_face_img2img:
                    print(f"# change checkpoint {checkpoint}")
                    api.util_set_model(checkpoint)

            if "seed" in frame_config:
                if frame_config["seed"] == -1:
                    seed = random.randrange(1, 2**63)
                else:
                    seed = frame_config["seed"]
                print(f"# seed {seed}")

            if "break" in frame_config:
                break

            if "base_prompt" in frame_config or "base_prompt2" in frame_config or "base_prompt3" in frame_config or "base_prompt4" in frame_config or "base_prompt5" in frame_config:
                base_prompt = config.base_prompt + "," + config.base_prompt2 + "," + config.base_prompt3 + "," + config.base_prompt4 + "," + config.base_prompt5

            for key in schedule_availables:
                if key in frame_config:
                    setattr(config, key, frame_config[key])

        frame_width = config.frame_width
        frame_height = config.frame_height

        input_img = Image.open(input_images_path_list[frame_index])
        input_img_arr = np.array(input_img)

        # fit frame size
        if input_img.width != frame_width or input_img.height != frame_height:
            input_img_arr = resize_image(input_img_arr, frame_width, frame_height, config.frame_resize, config.frame_resize_anchor)
            input_img = Image.fromarray(input_img_arr)

        # resume frame
        if frame_number < resume_frame:
            if frame_number == resume_frame - 1:
                last_image_arr = np.array(Image.open(output_image_path))
            continue

        if not overwrite:
            if os.path.isfile(output_image_path):
                print("skip")
                if frame_number < total_frames and not os.path.isfile(os.path.join(output_folder, os.path.basename(input_images_path_list[frame_index + 1]))):
                    print(f"last image {input_images_path_list[frame_index]}")
                    last_image_arr = np.array(Image.open(os.path.join(output_folder, os.path.basename(input_images_path_list[frame_index]))))
                continue

        ########################
        # base img2img
        if config.use_interrogate:
            ret = api.interrogate(input_img, config.interrogate_model)
            print(f"[interrogate({config.interrogate_model})] {ret.info}")
            interrogate_prompt = ret.info
        else:
            interrogate_prompt = ""

        prompt = base_prompt

        if config.use_base_img2img:
            for unit in controlnet_units:
                unit.input_image = input_img

            p_controlnet_units = []
            if config.start_frame == frame_number or (config.temporalnet_reset_frames != None and frame_number in config.temporalnet_reset_frames):
                if config.temporalnet_reset_interrogate:
                    ret = api.interrogate(input_img, config.interrogate_model)
                    print(f"[temporalnet reset interrogate({config.interrogate_model})] {ret.info}")
                    interrogate_prompt = ret.info

                p_controlnet_units = controlnet_units
            else:
                unit_tempo = None
                if config.temporalnet == "v2":
                    flow_image_arr = make_flow(input_images_path_list[frame_index - 1], input_images_path_list[frame_index], frame_width, frame_height, flow_image_folder, output_filename)
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
                prompt=prompt + interrogate_prompt,
                negative_prompt=config.neg_prompt,
                sampler_name=config.sampler_name,
                steps=config.sampler_step,
                images=[input_img],
                denoising_strength=config.denoising_strength,
                seed=-1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + frame_index,
                subseed=-1 if config.subseed_mode == "random" else subseed if config.subseed_mode == "fixed" else subseed + frame_index,
                subseed_strength=config.subseed_strength,
                cfg_scale=config.cfg_scale,
                initial_noise_multiplier=config.initial_noise_multiplier,
                width=frame_width,
                height=frame_height,
                controlnet_units=[x for x in p_controlnet_units if x is not None],
            )

            base_output_image = ret.images[0]
            base_output_image_arr = np.array(base_output_image)
            base_output_image.save(output_image_path)
        else:
            if config.temporalnet == "v2":
                if config.start_frame < frame_number:
                    flow_image_arr = make_flow(input_images_path_list[frame_index - 1], input_images_path_list[frame_index], frame_width, frame_height, flow_image_folder, output_filename)

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
            (zoom_images, zoom_coords, zoom_mask_arrs) = zoom_process(
                frame_index, input_img_arr, config.zoom_rects, config.zoom_blur, config.zoom_area_limit, config.zoom_max_resolusion, zoom_image_folder, output_filename
            )

            for zoom_index, (zoom_img, zoom_coord, zoom_mask_arr) in enumerate(zip(zoom_images, zoom_coords, zoom_mask_arrs)):
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
                    seed=-1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + frame_index,
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

                base_output_image_arr = merge_image(base_output_image_arr, zoom_output_image_arr, (x, y, calc_w, calc_h), zoom_mask_arr)

            if len(zoom_images) > 0:
                output_full_image_path = os.path.join(zoom_image_folder, output_filename)
                copyfile(output_image_path, output_full_image_path)
                base_output_image = Image.fromarray(base_output_image_arr)
                base_output_image.save(output_image_path)

        #####################
        # face img2img
        if config.use_face_img2img:
            if config.face_source == "input":
                (face_imgs, face_coords, face_mask_arrs) = face_process(input_img_arr, config.face_threshold, config.face_padding, config.face_blur, face_image_folder, output_filename)
            else:
                (face_imgs, face_coords, face_mask_arrs) = face_process(base_output_image_arr, config.face_threshold, config.face_padding, config.face_blur, face_image_folder, output_filename)

            # if len(config.face_selection):
            #     new_face_imgs = []
            #     new_face_coords = []
            #     new_face_mask_arrs = []
            #     for face_index in config.face_selection:
            #         if face_index < len(face_coords):
            #             if face_index not in last_face_coords:
            #                 new_face_imgs.append(face_imgs[face_index])
            #                 new_face_coords.append(face_coords[face_index])
            #                 new_face_mask_arrs.append(face_mask_arrs[face_index])
            #                 last_face_coords[face_index] = face_coords[face_index]
            #             else:
            #                 (x, y, w, h) = intersect_rect(face_coords[face_index], last_face_coords[face_index])
            #                 if w > 100 and h > 100:
            #                     new_face_imgs.append(face_imgs[face_index])
            #                     new_face_coords.append(face_coords[face_index])
            #                     new_face_mask_arrs.append(face_mask_arrs[face_index])
            #                     last_face_coords[face_index] = face_coords[face_index]

            #     (face_imgs, face_coords, face_mask_arrs) = (new_face_imgs, new_face_coords, new_face_mask_arrs)

            for face_index, (face_img, face_coord, face_mask_arr) in enumerate(zip(face_imgs, face_coords, face_mask_arrs)):
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
                    seed=-1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + frame_index,
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

                base_output_image_arr = merge_image(base_output_image_arr, face_output_image_arr, face_coord, face_mask_arr)

            if len(face_imgs) > 0:
                output_full_image_path = os.path.join(face_image_folder, output_filename)
                copyfile(output_image_path, output_full_image_path)
                base_output_image = Image.fromarray(base_output_image_arr)
                base_output_image.save(output_image_path)

        base_output_image = base_output_image_arr
        last_image_arr = base_output_image_arr

        if end_frame > 0 and end_frame == frame_number:
            break
