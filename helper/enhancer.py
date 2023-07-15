import os
import glob
import helper.webuiapi as webuiapi
from PIL import Image
import numpy as np
from helper.temporalnet2 import make_flow
from helper.temporalnet2 import encode_image
from helper.zoom import process as zoom_process
from helper.facedetect import merge_face, process as face_process
from helper.image_util import zoom_image, resize_image, crop_and_resize
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


def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


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


def run(
    project_folder="",
    init_image_path="",
    input_folder="./input",
    output_folder="./output",
    zoom_image_folder="./zoom_images",
    face_image_folder="./face_images",
    flow_image_folder="./flow_images",
    base_prompt="",
    face_prompt="",
    neg_prompt="",
    checkpoint="",
    seed=-1,
    seed_mode="fixed",
    sampler_name="Euler a",
    sampler_step=30,
    face_sampler_step=30,
    cfg_scale=5,
    frame_width=720,
    frame_height=1280,
    frame_crop=None,
    frame_zoom=None,
    interrogate_model="clip",
    face_interrogate_model="deepdanbooru",
    use_base_img2img=True,
    use_zoom_img2img=False,
    use_face_img2img=False,
    copy_base_image="input",
    denoising_strength=0.75,
    face_denoising_strength=0.5,
    face_threshold=0.2,
    face_padding=16,
    zoom_denoising_strength=0.5,
    temporalnet_reset_frames=[],
    temporalnet="v2",
    temporalnet_weight=0.4,
    zoom_temporalnet="v2",
    zoom_temporalnet_weight=0.4,
    face_temporalnet="v2",
    face_temporalnet_weight=0.4,
    use_interrogate=False,
    use_face_interrogate=False,
    # zoom_area_limit: 2048*1080,
    zoom_area_limit=1280 * 1024,
    # zoom_max_resolusion: 2048,
    zoom_max_resolusion=1280,
    # 0, 400, 1080, 512, 21, 999999
    zoom_rects=[],
    controlnet_lowvram=False,
    controlnet=[],
    zoom_controlnet=[],
    face_controlnet=[],
    frame_schedule={},
    chromakey="",
    resume_index=0,
    config={},
):
    print(f"# project path {project_folder}")

    if not input_folder and not input_video_path:
        print(f"# input error")
        return

    if project_folder:
        input_folder = os.path.normpath(os.path.join(project_folder, input_folder))
        output_folder = os.path.normpath(os.path.join(project_folder, output_folder))
        zoom_image_folder = os.path.normpath(os.path.join(project_folder, zoom_image_folder))
        face_image_folder = os.path.normpath(os.path.join(project_folder, face_image_folder))
        flow_image_folder = os.path.normpath(os.path.join(project_folder, flow_image_folder))

    print(f"# input images path {input_folder}")
    print(f"# output folder {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(zoom_image_folder, exist_ok=True)
    os.makedirs(face_image_folder, exist_ok=True)
    os.makedirs(flow_image_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"# extract input_video_path")

    if seed == -1:
        seed = random.randrange(1, 2**63)
    print(f"# seed {seed}")
    print(f"# seed mode {seed_mode}")

    if checkpoint:
        api.refresh_checkpoints()
        print(f"# change checkpoint {checkpoint}")
        api.util_set_model(checkpoint)

    current_model = api.util_get_current_model()
    print(f"# current checkpoint {current_model}")

    controlnet_units = []
    for cn in controlnet:
        cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=controlnet_lowvram)
        controlnet_units.append(cn_unit)

    zoom_controlnet_units = []
    for cn in zoom_controlnet:
        cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=controlnet_lowvram)
        zoom_controlnet_units.append(cn_unit)

    face_controlnet_units = []
    for cn in face_controlnet:
        cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=controlnet_lowvram)
        face_controlnet_units.append(cn_unit)

    init_image = None
    if init_image_path:
        init_image = Image.open(os.path.join(project_folder, init_image_path))

    input_images_path_list = get_image_paths(input_folder)

    input_img = None
    input_img_arr = None

    last_image = None
    last_image_arr = None
    flow_image_arr = None

    base_output_image = None
    base_output_image_arr = None

    start_index = 0
    if init_image != None:
        last_image = init_image
        last_image_arr = np.array(init_image)
        start_index = 1

    if resume_index != 0:
        start_index = resume_index
        last_image = Image.open(input_images_path_list[resume_index - 1])
        last_image_arr = np.array(last_image)
        if last_image.width != frame_width or last_image.height != frame_height:
            last_image_arr = resize_image(last_image_arr, frame_width, frame_height)

    current_zoom_scale = 1
    current_zoom_offset_x = 0
    current_zoom_offset_y = 0
    current_zoom_step = None
    goal_zoom_scale = None
    goal_zoom_offset_x = None
    goal_zoom_offset_y = None
    goal_zoom_step = None

    for i in range(start_index, len(input_images_path_list)):
        frame_number = i + 1
        print(f"# frame {frame_number}")

        #####################
        # frame schedule
        if frame_number in frame_schedule:
            print(f"# frame {frame_number}")
            frame_config = frame_schedule[frame_number]
            if "checkpoint" in frame_config:
                checkpoint = frame_config["checkpoint"]
                print(f"# change checkpoint {checkpoint}")
                api.util_set_model(checkpoint)

            if "rollback" in frame_config:
                frame_config = config
                print(f"# rollback")

            if "break" in frame_config:
                break

            for key in schedule_availables:
                if key in frame_config:
                    locals()[key] = frame_config[key]

        output_filename = os.path.basename(input_images_path_list[i])
        output_image_path = os.path.join(output_folder, output_filename)

        current_frame_width = frame_width
        current_frame_height = frame_height

        input_img = Image.open(input_images_path_list[i])
        input_img_arr = np.array(input_img)

        # zoom scale
        if frame_zoom != None:
            if isinstance(frame_zoom, list):
                if len(frame_zoom) == 4:
                    [scale, offset_x, offset_y, frames] = frame_zoom
                if len(frame_zoom) == 3:
                    [scale, offset_x, offset_y] = frame_zoom
                    frames = 0
                if len(frame_zoom) == 1:
                    [scale] = frame_zoom
                    offset_x = 0
                    offset_y = 0
                    frames = 0
            else:
                scale = frame_zoom
                offset_x = 0
                offset_y = 0
                frames = 0
            if frames > 0:
                goal_zoom_scale = scale
                goal_zoom_offset_x = offset_x
                goal_zoom_offset_y = offset_y
                goal_zoom_step = frames
                current_zoom_scale = 1
                current_zoom_offset_x = 0
                current_zoom_offset_y = 0
                current_zoom_step = 1
            else:
                goal_zoom_step = 0
                current_zoom_scale = scale
                current_zoom_offset_x = offset_x
                current_zoom_offset_y = offset_y
                current_zoom_step = 0
            frame_zoom = None

        if goal_zoom_scale != None:
            t = current_zoom_step / goal_zoom_step
            print(f"t {t}")
            current_zoom_scale = (1 - t) + t * goal_zoom_scale
            current_zoom_offset_x = (int)((1 - t) + t * goal_zoom_offset_x)
            current_zoom_offset_y = (int)((1 - t) + t * goal_zoom_offset_y)

        if current_zoom_scale != None and current_zoom_scale > 1:
            print(f"zoom scale {current_zoom_scale} ({current_zoom_offset_x},{current_zoom_offset_y})")
            input_img_arr = zoom_image(input_img_arr, current_zoom_scale)
            (h, w) = input_img_arr.shape[:2]
            x = ((w - input_img.width) >> 1) + current_zoom_offset_x
            y = ((h - input_img.height) >> 1) + current_zoom_offset_y
            input_img_arr = input_img_arr[y : y + current_frame_height, x : x + current_frame_width]
            input_img = Image.fromarray(input_img_arr)
            current_zoom_step = current_zoom_step + 1
            if current_zoom_step > goal_zoom_step:
                goal_zoom_scale = None

        if frame_crop != None and len(frame_crop) == 4:
            [x, y, w, h] = frame_crop
            input_img_arr = input_img_arr[y : y + h, x : x + w]
            input_img = Image.fromarray(input_img_arr)
            # input_img.save(os.path.join(output_folder, output_filename))

        if input_img.width != frame_width or input_img.height != frame_height:
            input_img_arr = resize_image(input_img_arr, frame_width, frame_height, "crop")
            input_img = Image.fromarray(input_img_arr)

        if use_interrogate:
            ret = api.interrogate(input_img, interrogate_model)
            print(f"[interrogate({interrogate_model})] {ret.info}")
            prompt = base_prompt + ret.info
        else:
            prompt = base_prompt

        ########################
        # base img2img
        if use_base_img2img:
            for unit in controlnet_units:
                unit.input_image = input_img

            p_controlnet_units = []
            if i == 0 or frame_number in temporalnet_reset_frames:
                p_controlnet_units = controlnet_units
            else:
                unit_tempo = None
                if temporalnet == "v2":
                    flow_image_arr = make_flow(input_images_path_list[i - 1], input_images_path_list[i], current_frame_width, current_frame_height, flow_image_folder, output_filename)
                    unit_tempo = unit_tempo_v2
                    unit_tempo.weight = temporalnet_weight
                    unit_tempo.encoded_image = encode_image(flow_image_arr, last_image_arr)
                    unit_tempo.lowvram = controlnet_lowvram
                elif temporalnet == "v1":
                    unit_tempo = unit_tempo_v1
                    unit_tempo.weight = temporalnet_weight
                    unit_tempo.input_image = last_image
                    unit_tempo.lowvram = controlnet_lowvram
                p_controlnet_units = controlnet_units + [unit_tempo]

            ret = api.img2img(
                prompt=prompt,
                negative_prompt=neg_prompt,
                sampler_name=sampler_name,
                steps=sampler_step,
                images=[input_img],
                denoising_strength=denoising_strength,
                seed=-1 if seed_mode == "random" else seed if seed_mode == "fixed" else seed + i,
                cfg_scale=cfg_scale,
                width=current_frame_width,
                height=current_frame_height,
                controlnet_units=[x for x in p_controlnet_units if x is not None],
            )

            base_output_image = ret.images[0]
            base_output_image_arr = np.array(base_output_image)
            base_output_image.save(output_image_path)
        else:
            if copy_base_image == "base":
                if os.path.exists(os.path.join(zoom_image_folder, output_filename)):
                    base_output_image = Image.open(os.path.join(zoom_image_folder, output_filename))
                    base_output_image_arr = np.array(base_output_image)
                elif os.path.exists(os.path.join(face_image_folder, output_filename)):
                    base_output_image = Image.open(os.path.join(face_image_folder, output_filename))
                    base_output_image_arr = np.array(base_output_image)
                elif os.path.exists(output_image_path):
                    base_output_image = Image.open(output_image_path)
                    base_output_image_arr = np.array(base_output_image)

            else:
                base_output_image = input_img
                base_output_image_arr = np.array(base_output_image)
                base_output_image.save(output_image_path)

        ######################
        # zoom img2img
        if use_zoom_img2img:
            (zoom_rects, zoom_images, zoom_coords, masks) = zoom_process(i, input_img_arr, zoom_rects, zoom_area_limit, zoom_max_resolusion, zoom_image_folder, output_filename)

            for zoom_index, (zoom_img, zoom_coord, mask) in enumerate(zip(zoom_images, zoom_coords, masks)):
                [x, y, re_w, re_h, calc_w, calc_h] = zoom_coord

                for unit in zoom_controlnet_units:
                    unit.input_image = zoom_img

                p_zoom_controlnet_units = []
                if i == 0 or frame_number in temporalnet_reset_frames:
                    p_zoom_controlnet_units = zoom_controlnet_units
                else:
                    unit_tempo = None
                    last_zoom_img_arr = crop_and_resize(last_image_arr, [x, y, calc_w, calc_h], zoom_img.width, zoom_img.height)
                    if zoom_temporalnet == "v2":
                        flow_zoom_image_arr = crop_and_resize(flow_image_arr, [x, y, calc_w, calc_h], zoom_img.width, zoom_img.height)
                        unit_tempo = unit_tempo_v2
                        unit_tempo.weight = zoom_temporalnet_weight
                        unit_tempo.encoded_image = encode_image(flow_zoom_image_arr, last_zoom_img_arr)
                        unit_tempo.lowvram = controlnet_lowvram

                    elif zoom_temporalnet == "v1":
                        unit_tempo = unit_tempo_v1
                        unit_tempo.weight = zoom_temporalnet_weight
                        unit_tempo.input_image = Image.fromarray(last_zoom_img_arr)
                        unit_tempo.lowvram = controlnet_lowvram
                    p_zoom_controlnet_units = zoom_controlnet_units + [unit_tempo]
                print(f" * zoom {zoom_index}")

                ret = api.img2img(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    sampler_name=sampler_name,
                    steps=sampler_step,
                    images=[zoom_img],
                    denoising_strength=zoom_denoising_strength,
                    seed=-1 if seed_mode == "random" else seed if seed_mode == "fixed" else seed + i,
                    cfg_scale=cfg_scale,
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

                base_output_image_arr = merge_face(base_output_image_arr, zoom_output_image_arr, (x, y, calc_w, calc_h), mask)

            if len(zoom_images) > 0:
                output_full_image_path = os.path.join(zoom_image_folder, output_filename)
                copyfile(output_image_path, output_full_image_path)
                base_output_image = Image.fromarray(base_output_image_arr)
                base_output_image.save(output_image_path)

        #####################
        # face img2img
        if use_face_img2img:
            (face_imgs, new_coords, masks) = face_process(input_img_arr, face_threshold, face_padding, face_image_folder, output_filename)

            for face_index, (face_img, new_coord, mask) in enumerate(zip(face_imgs, new_coords, masks)):
                for unit in face_controlnet_units:
                    unit.input_image = face_img

                p_face_controlnet_units = []
                if i == 0 or frame_number in temporalnet_reset_frames:
                    p_face_controlnet_units = face_controlnet_units
                else:
                    unit_tempo = None
                    last_face_img_arr = crop_and_resize(last_image_arr, new_coord, face_img.width, face_img.height)
                    if face_temporalnet == "v2":
                        flow_face_image_arr = crop_and_resize(flow_image_arr, new_coord, face_img.width, face_img.height)
                        unit_tempo = unit_tempo_v2
                        unit_tempo.weight = face_temporalnet_weight
                        unit_tempo.lowvram = controlnet_lowvram
                        unit_tempo.encoded_image = encode_image(flow_face_image_arr, last_face_img_arr)
                    elif face_temporalnet == "v1":
                        unit_tempo = unit_tempo_v1
                        unit_tempo.weight = face_temporalnet_weight
                        unit_tempo.input_image = Image.fromarray(last_face_img_arr)
                        unit_tempo.lowvram = controlnet_lowvram
                    p_face_controlnet_units = face_controlnet_units + [unit_tempo]

                if use_face_interrogate:
                    ret = api.interrogate(face_img, face_interrogate_model)
                    print(f"[face-interrogate({face_interrogate_model})] {ret.info}")
                    prompt = face_prompt + ret.info
                else:
                    prompt = face_prompt
                print(f" * face {face_index}")

                ret = api.img2img(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    sampler_name=sampler_name,
                    steps=face_sampler_step,
                    images=[face_img],
                    denoising_strength=face_denoising_strength,
                    seed=-1 if seed_mode == "random" else seed if seed_mode == "fixed" else seed + i,
                    cfg_scale=cfg_scale,
                    width=face_img.width,
                    height=face_img.height,
                    controlnet_units=[x for x in p_face_controlnet_units if x is not None],
                )

                output_face_filename = f"{os.path.splitext(output_filename)[0]}-face-convert{face_index}.png"
                output_face_image_path = os.path.join(face_image_folder, output_face_filename)
                face_output_image = ret.images[0]
                face_output_image.save(output_face_image_path)
                face_output_image_arr = np.array(face_output_image)

                base_output_image_arr = merge_face(base_output_image_arr, face_output_image_arr, new_coord, mask)

            if len(face_imgs) > 0:
                output_full_image_path = os.path.join(face_image_folder, output_filename)
                copyfile(output_image_path, output_full_image_path)
                base_output_image = Image.fromarray(base_output_image_arr)
                base_output_image.save(output_image_path)

        base_output_image = base_output_image_arr
        last_image_arr = base_output_image_arr
