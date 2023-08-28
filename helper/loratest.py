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

api = webuiapi.WebUIApi()

def run(config: Config, project_folder: str):
    print(f"# project path {project_folder}")

    if project_folder:
        input_folder = os.path.normpath(os.path.join(project_folder, config.input_folder))
        output_folder = input_folder + "_loratest"
        zoom_image_folder = output_folder
        face_image_folder = output_folder
    else:
        print("project path not found")
        return

    print(f"# input images path {input_folder}")
    print(f"# output folder {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

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

    if config.checkpoint:
        if config.use_base_img2img or config.use_zoom_img2img or config.use_face_img2img:
            api.refresh_checkpoints()
            print(f"# change checkpoint {config.checkpoint}")
            api.util_set_model(config.checkpoint)
    
    loratest_list = get_lora_paths(loratest)

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

    input_images_path_list = get_image_paths(input_folder)

    input_img = None
    input_img_arr = None

    base_output_image = None
    base_output_image_arr = None

    start_index = 0

    if not config.start_frame:
        config.start_frame = 1

    total_frames = len(input_images_path_list)

    base_prompt = config.base_prompt+","+config.base_prompt2+","+config.base_prompt3+","+config.base_prompt4+","+config.base_prompt5
    interrogate_prompt = ""

    for frame_index in range(start_index, total_frames):
        if samplerun_index == len(loratest_list):
            break
        print(loratest_list[frame_index])
        loratest = os.path.splitext(os.path.basename(loratest_list[frame_index]))[0]
        output_filename = loratest + ".png"
        print(f"loratest output_filename {output_filename}")
        output_image_path = os.path.join(output_folder, output_filename)
        samplerun_index = samplerun_index + 1
        frame_index = 0

        frame_number = frame_index + 1
        print(f"# frame {frame_number}/{total_frames}")

        if config.start_frame > frame_number:
            continue

        frame_width = config.frame_width
        frame_height = config.frame_height

        input_img = Image.open(input_images_path_list[frame_index])
        input_img_arr = np.array(input_img)

        # fit frame size
        if input_img.width != frame_width or input_img.height != frame_height:
            input_img_arr = resize_image(input_img_arr, frame_width, frame_height, config.frame_resize, config.frame_resize_anchor)
            input_img = Image.fromarray(input_img_arr)

        ########################
        # base img2img
        if config.use_interrogate:
            ret = api.interrogate(input_img, config.interrogate_model)
            print(f"[interrogate({config.interrogate_model})] {ret.info}")
            interrogate_prompt = ret.info
        else:
            interrogate_prompt = ""

        prompt = base_prompt
        prompt = prompt + f" <lora:{loratest}:0.7>"

        if config.use_base_img2img:
            for unit in controlnet_units:
                unit.input_image = input_img

            p_controlnet_units = []
            if config.temporalnet_reset_interrogate:
                ret = api.interrogate(input_img, config.interrogate_model)
                print(f"[temporalnet reset interrogate({config.interrogate_model})] {ret.info}")
                interrogate_prompt = ret.info

            p_controlnet_units = controlnet_units

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
                p_zoom_controlnet_units = zoom_controlnet_units
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

            for face_index, (face_img, face_coord, face_mask_arr) in enumerate(zip(face_imgs, face_coords, face_mask_arrs)):
                for unit in face_controlnet_units:
                    unit.input_image = face_img

                p_face_controlnet_units = []
                p_face_controlnet_units = face_controlnet_units

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
