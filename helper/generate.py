import os
import helper.webuiapi as webuiapi
from PIL import Image
import numpy as np
from helper.temporalnet2 import make_flow, encode_image
from helper.config import Config
from helper.util import get_image_paths
import random


schedule_availables = [
    "base_prompt",
    "seed",
    "seed_mode",
    "generate_sampler_name",
    "generate_sampler_step",
    "cfg_scale",
    "generate_width",
    "generate_height",
    "temporalnet",
    "temporalnet_weight",
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


def run(config: Config, project_folder: str, overwrite: bool, resume_frame: int, start_frame: int, end_frame: int):
    print(f"# project path {project_folder}")

    if project_folder:
        input_folder = os.path.normpath(os.path.join(project_folder, config.generate_input_folder))
        output_folder = os.path.normpath(os.path.join(project_folder, config.generate_output_folder))
        flow_image_folder = os.path.normpath(os.path.join(output_folder, "./flow_images"))
    else:
        print("project path not found")
        return

    print(f"# input images path {input_folder}")
    print(f"# input generate folder {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
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

    if config.checkpoint:
        api.refresh_checkpoints()
        print(f"# change checkpoint {config.checkpoint}")
        api.util_set_model(config.checkpoint)

    controlnet_units = []
    for cn in config.generate_controlnet:
        cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=config.controlnet_lowvram)
        controlnet_units.append(cn_unit)

    input_images_path_list = get_image_paths(input_folder)

    input_img = None

    last_image_arr = None
    flow_image_arr = None

    base_output_image = None
    base_output_image_arr = None

    start_index = 0

    total_frames = len(input_images_path_list)

    base_prompt = config.base_prompt + "," + config.base_prompt2 + "," + config.base_prompt3 + "," + config.base_prompt4 + "," + config.base_prompt5
    interrogate_prompt = ""

    for frame_index in range(start_index, total_frames):
        output_filename = os.path.basename(input_images_path_list[frame_index])
        output_image_path = os.path.join(output_folder, output_filename)

        frame_number = frame_index + 1
        print(f"# frame {frame_number}/{total_frames}")

        if start_frame > frame_number:
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

        input_img = Image.open(input_images_path_list[frame_index])

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
        # txt2img
        if config.use_interrogate:
            ret = api.interrogate(input_img, config.interrogate_model)
            print(f"[interrogate({config.interrogate_model})] {ret.info}")
            interrogate_prompt = ret.info
        else:
            interrogate_prompt = ""

        for unit in controlnet_units:
            unit.input_image = input_img

        p_controlnet_units = []
        if start_frame == frame_number or (config.temporalnet_reset_frames != None and frame_number in config.temporalnet_reset_frames):
            if config.temporalnet_reset_interrogate:
                ret = api.interrogate(input_img, config.interrogate_model)
                print(f"[temporalnet reset interrogate({config.interrogate_model})] {ret.info}")
                interrogate_prompt = ret.info

            p_controlnet_units = controlnet_units
        else:
            unit_tempo = None
            if config.temporalnet == "v2":
                flow_image_arr = make_flow(
                    input_images_path_list[frame_index - 1], input_images_path_list[frame_index], config.generate_width, config.generate_height, flow_image_folder, output_filename
                )
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

        if config.generate_hiresfix:
            ret = api.txt2img(
                prompt=base_prompt + interrogate_prompt,
                negative_prompt=config.neg_prompt,
                sampler_name=config.generate_sampler_name,
                steps=config.generate_sampler_step,
                seed=-1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + frame_index,
                subseed=-1 if config.subseed_mode == "random" else subseed if config.subseed_mode == "fixed" else subseed + frame_index,
                subseed_strength=config.subseed_strength,
                cfg_scale=config.cfg_scale,
                width=config.generate_width,
                height=config.generate_height,
                enable_hr=True,
                hr_scale=2,
                hr_upscaler="4x-UltraSharp",
                hr_second_pass_steps=config.generate_sampler_step / 2,
                denoising_strength=0.4,
                controlnet_units=[x for x in p_controlnet_units if x is not None],
            )
        else:
            ret = api.txt2img(
                prompt=base_prompt + interrogate_prompt,
                negative_prompt=config.neg_prompt,
                sampler_name=config.generate_sampler_name,
                steps=config.generate_sampler_step,
                seed=-1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + frame_index,
                subseed=-1 if config.subseed_mode == "random" else subseed if config.subseed_mode == "fixed" else subseed + frame_index,
                subseed_strength=config.subseed_strength,
                cfg_scale=config.cfg_scale,
                width=config.generate_width,
                height=config.generate_height,
                controlnet_units=[x for x in p_controlnet_units if x is not None],
            )

        base_output_image = ret.images[0]
        base_output_image_arr = np.array(base_output_image)
        base_output_image.save(output_image_path)

        base_output_image = base_output_image_arr
        last_image_arr = base_output_image_arr

        if end_frame > 0 and end_frame == frame_number:
            break
