import os
import helper.webuiapi as webuiapi
from PIL import Image
import numpy as np
from helper.temporalnet2 import make_flow, encode_image
from helper.image_util import resize_image
from helper.config import Config
from helper.util import get_image_paths
import random

schedule_availables = [
    "base_prompt",
    "base_prompt2",
    "base_prompt3",
    "base_prompt4",
    "base_prompt5",
    "seed",
    "seed_mode",
    "sampler_name",
    "sampler_step",
    "cfg_scale",
    "use_base_img2img",
    "denoising_strength",
    "use_interrogate",
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
unit_tempo_v1xl = webuiapi.ControlNetUnit(
    module="none",
    model="temporalnet-sdxl-1.0 [7b9fb926]",
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

alwayson_scripts = {}
freeu_sd15 = {
    "args": [{
        "enable": True,
        "start_ratio": 0.0,
        "stop_ratio": 1.0,
        "transition_smoothness": 0.0,
        "stage_infos": [
            {
                "backbone_factor": 1.2,
                "backbone_offset": 0.0,
                "backbone_width": 0.5,
                "skip_factor": 0.9,
                "skip_high_end_factor": 1.0,
                "skip_cutoff": 0.0
            },
            {
                "backbone_factor": 1.4,
                "backbone_offset": 0.0,
                "backbone_width": 0.5,
                "skip_factor": 0.2,
                "skip_high_end_factor": 1.0,
                "skip_cutoff": 0.0
            },
            {
                "backbone_factor": 1.0,
                "backbone_offset": 0.0,
                "backbone_width": 0.5,
                "skip_factor": 1.0,
                "skip_high_end_factor": 1.0,
                "skip_cutoff": 0.0
            }
        ]
    }]
}
freeu_sdxl = {
    "args": [{
        "enable": True,
        "start_ratio": 0.0,
        "stop_ratio": 1.0,
        "transition_smoothness": 0.0,
        "stage_infos": [
            {
                "backbone_factor": 1.1,
                "backbone_offset": 0.0,
                "backbone_width": 0.5,
                "skip_factor": 0.6,
                "skip_high_end_factor": 1.0,
                "skip_cutoff": 0.0
            },
            {
                "backbone_factor": 1.2,
                "backbone_offset": 0.0,
                "backbone_width": 0.5,
                "skip_factor": 0.4,
                "skip_high_end_factor": 1.0,
                "skip_cutoff": 0.0
            },
            {
                "backbone_factor": 1.0,
                "backbone_offset": 0.0,
                "backbone_width": 0.5,
                "skip_factor": 1.0,
                "skip_high_end_factor": 1.0,
                "skip_cutoff": 0.0
            }
        ]
    }]
}

def run(config: Config, project_folder: str, overwrite: bool, reverse: bool, resume_frame: int, start_frame: int, end_frame: int, rework_mode: str = None):
    if config.host != "":
        api = webuiapi.WebUIApi(host=config.host, port=config.port, use_https=config.https, timeout=120)
    else:
        api = webuiapi.WebUIApi()

    print(f"# project path {project_folder}")

    if project_folder:
        input_folder = os.path.normpath(os.path.join(project_folder, config.input_folder))
        if config.mask_folder != "":
            mask_folder = os.path.normpath(os.path.join(project_folder, config.mask_folder))
        else:
            mask_folder = ""
        output_folder = os.path.normpath(os.path.join(project_folder, config.output_folder))
        zoom_image_folder = os.path.normpath(os.path.join(output_folder, "./zoom_images"))
        flow_image_folder = os.path.normpath(os.path.join(output_folder, "./flow_images"))
    else:
        print("project path not found")
        return

    print(f"# input images path {input_folder}")
    print(f"# mask images path {mask_folder}")
    print(f"# output folder {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(zoom_image_folder, exist_ok=True)
    os.makedirs(flow_image_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"# not found input_video_path")

    if config.temporalnet == "v1xl":
        alwayson_scripts["freeu"] = freeu_sdxl
    else:
        alwayson_scripts["freeu"] = freeu_sd15

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
        if config.use_base_img2img or config.use_zoom_img2img:
            api.refresh_checkpoints()
            print(f"# change checkpoint {config.checkpoint}")
            api.util_set_model(config.checkpoint)

    controlnet_units = []
    for cn in config.controlnet:
        cn_unit = webuiapi.ControlNetUnit(**cn, lowvram=config.controlnet_lowvram)
        controlnet_units.append(cn_unit)

    init_image = None
    if config.init_image_path:
        init_image = Image.open(os.path.join(project_folder, config.init_image_path))

    if mask_folder != "":
        mask_images_path_list = get_image_paths(mask_folder)
        if reverse:
            mask_images_path_list.reverse()

    input_images_path_list = get_image_paths(input_folder)
    if reverse:
        input_images_path_list.reverse()

    input_img = None
    input_img_arr = None

    last_image_arr = None
    flow_image_arr = None

    base_output_image = None
    base_output_image_arr = None

    start_index = 0
    if init_image != None:
        last_image_arr = np.array(init_image)

    total_frames = len(input_images_path_list)
    print(f"total frames {total_frames}")

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
                if config.use_base_img2img:
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
            
            print(f"{frame_config=}")
            if "base_prompt" in frame_config:
                config.base_prompt = frame_config["base_prompt"]
            if "base_prompt2" in frame_config:
                config.base_prompt2 = frame_config["base_prompt2"]
            if "base_prompt3" in frame_config:
                config.base_prompt3 = frame_config["base_prompt3"]
            if "base_prompt4" in frame_config:
                config.base_prompt4 = frame_config["base_prompt4"]
            if "base_prompt5" in frame_config:
                config.base_prompt5 = frame_config["base_prompt5"]
            if "base_prompt" in frame_config or "base_prompt2" in frame_config or "base_prompt3" in frame_config or "base_prompt4" in frame_config or "base_prompt5" in frame_config:
                base_prompt = config.base_prompt + "," + config.base_prompt2 + "," + config.base_prompt3 + "," + config.base_prompt4 + "," + config.base_prompt5

            for key in schedule_availables:
                if key in frame_config:
                    setattr(config, key, frame_config[key])

        frame_width = config.frame_width
        frame_height = config.frame_height

        input_img = Image.open(input_images_path_list[frame_index])
        input_img_arr = np.array(input_img)

        mask_img = None
        if mask_folder != "":
            mask_img = Image.open(mask_images_path_list[frame_index]).convert("RGB")

        # fit frame size
        if input_img.width != frame_width or input_img.height != frame_height:
            input_img_arr = resize_image(input_img_arr, frame_width, frame_height, config.frame_resize, config.frame_resize_anchor)
            input_img = Image.fromarray(input_img_arr)

        if mask_img != None:
            if mask_img.width != frame_width or mask_img.height != frame_height:
                mask_img = Image.fromarray(resize_image(np.array(mask_img), frame_width, frame_height, config.frame_resize, config.frame_resize_anchor))

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

        if config.use_interrogate:
            ret = api.interrogate(input_img, config.interrogate_model)
            print(f"[interrogate({config.interrogate_model})] {ret.info}")
            interrogate_prompt = ret.info
        else:
            interrogate_prompt = ""

        prompt = base_prompt

        if config.use_base_img2img:
            for unit in controlnet_units:
                if unit.input_image_path != None:
                    unit.input_image = Image.open(os.path.join(project_folder, unit.input_image_path))
                    print(f"input image {unit.module}")
                else:
                    unit.input_image = input_img

            p_controlnet_units = []
            use_temporalnet = True
            if start_frame == frame_number or (config.temporalnet_reset_frames != None and frame_number in config.temporalnet_reset_frames):
                if config.temporalnet_reset_interrogate:
                    ret = api.interrogate(input_img, config.interrogate_model)
                    print(f"[temporalnet reset interrogate({config.interrogate_model})] {ret.info}")
                    interrogate_prompt = ret.info

                p_controlnet_units = controlnet_units
                use_temporalnet = not config.temporalnet_loopback

            if use_temporalnet:
                unit_tempo = None
                if config.temporalnet == "v2":
                    flow_image_arr = make_flow(input_images_path_list[frame_index - 1], input_images_path_list[frame_index], frame_width, frame_height, flow_image_folder, output_filename)
                    unit_tempo = unit_tempo_v2
                    unit_tempo.weight = config.temporalnet_weight
                    unit_tempo.encoded_image = encode_image(flow_image_arr, last_image_arr if config.temporalnet_loopback else input_img_arr)
                    unit_tempo.lowvram = config.controlnet_lowvram
                elif config.temporalnet == "v1":
                    unit_tempo = unit_tempo_v1
                    unit_tempo.weight = config.temporalnet_weight
                    unit_tempo.input_image = Image.fromarray(last_image_arr if config.temporalnet_loopback else input_img_arr)
                    unit_tempo.lowvram = config.controlnet_lowvram
                elif config.temporalnet == "v1xl":
                    unit_tempo = unit_tempo_v1xl
                    unit_tempo.weight = config.temporalnet_weight
                    unit_tempo.input_image = Image.fromarray(last_image_arr if config.temporalnet_loopback else input_img_arr)
                    unit_tempo.lowvram = config.controlnet_lowvram
                p_controlnet_units = controlnet_units + [unit_tempo]

            current_seed = -1 if config.seed_mode == "random" else seed if config.seed_mode == "fixed" else seed + frame_index
            # print(f"{config.denoising_strength=}")
            override_settings = {}
            if config.vae != "":
                override_settings = {
                    "sd_vae": config.vae,
                }
            ret = api.img2img(
                prompt=prompt + interrogate_prompt,
                negative_prompt=config.neg_prompt,
                sampler_name=config.sampler_name,
                steps=config.sampler_step,
                images=[input_img],
                mask_image=mask_img,
                mask_blur=16,
                inpainting_fill=1,
                inpaint_full_res=False,
                inpaint_full_res_padding=4,
                denoising_strength=config.denoising_strength,
                seed=current_seed,
                subseed=-1 if config.subseed_mode == "random" else subseed if config.subseed_mode == "fixed" else subseed + frame_index,
                subseed_strength=config.subseed_strength,
                cfg_scale=config.cfg_scale,
                initial_noise_multiplier=config.initial_noise_multiplier,
                width=frame_width,
                height=frame_height,
                controlnet_units=[x for x in p_controlnet_units if x is not None],
                override_settings=override_settings,
                alwayson_scripts=alwayson_scripts
            )

            base_output_image = ret.images[0]
            base_output_image_arr = np.array(base_output_image)
            base_output_image.save(output_image_path)
        else:
            if config.temporalnet == "v2":
                if start_frame < frame_number:
                    flow_image_arr = make_flow(input_images_path_list[frame_index - 1], input_images_path_list[frame_index], frame_width, frame_height, flow_image_folder, output_filename)

            if rework_mode:
                if os.path.exists(output_image_path):
                    base_output_image = Image.open(output_image_path)
                    base_output_image_arr = np.array(base_output_image)

            else:
                base_output_image = input_img
                base_output_image_arr = np.array(base_output_image)
                base_output_image.save(output_image_path)

        last_image_arr = base_output_image_arr

        if end_frame > 0 and end_frame == frame_number:
            break
