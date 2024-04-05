import os
import json
import glob
from PIL import Image
import argparse
import helper.enhancer as enhancer
import helper.generate as generate
import helper.dynamic_zoom as dynamic_zoom
import helper.loratest as loratest
import helper.samplerun as samplerun
import helper.padding_end as padding_end
import helper.video_util as video_util
import helper.patch_cover as patch_cover
from helper.config import Config, PatchConfig


def get_args():
    parser = argparse.ArgumentParser()
    # run
    parser.add_argument("configfile", help="animatediff config")
    parser.add_argument("--openpose", dest="openpose", default=False)
    parser.add_argument("--tile", dest="tile", default=True)
    parser.add_argument("--strength", dest="strength", default=0.45)
    parser.add_argument("--rework", dest="rework", default=False)
    parser.add_argument("--prompt", dest="prompt", default="")
    parser.add_argument("--skip", dest="skip", default=False)
    parser.add_argument("--output", dest="output", default=None)

    return parser.parse_args()


args = get_args()

configpath = os.path.dirname(args.configfile)

with open(args.configfile, "r") as json_read:
    config = json.load(json_read)

con_config = Config()

if not "prompt_map" in config and "prompt" in config:
    config["prompt_map"] = {"0": config["prompt"][0]}

con_config.base_prompt = config["prompt_map"]["0"]
con_config.neg_prompt = config["n_prompt"][0]
con_config.seed = config["seed"][0]
con_config.seed_mode = "fixed"
con_config.cfg_scale = config["guidance_scale"]
if con_config.cfg_scale >= 9:
    con_config.cfg_scale = 9
if con_config.cfg_scale <= 5:
    con_config.cfg_scale = 5
if config["scheduler"] == "k_dpmpp_2m":
    sampler_name = "DPM++ 2M Karras"
elif config["scheduler"] == "k_dpmpp_sde":
    sampler_name = "DPM++ SDE Karras"
elif config["scheduler"] == "ddim":
    sampler_name = "DDIM"
else:
    sampler_name = "Euler a"
con_config.sampler_name = sampler_name
con_config.sampler_step = config["steps"]

anim_name = glob.glob(os.path.join(configpath, "*.gif"))[0]
base_name = os.path.splitext(os.path.basename(anim_name))[0]
base_name = "_".join(base_name.split("_")[0:2])
base_name = base_name.replace("_", "-")
print(base_name)
image = Image.open(anim_name)
print(f"{image.width=} {image.height=}")

con_config.frame_width = image.width * 2
con_config.frame_height = image.height * 2
con_config.input_folder = "./" + base_name
if args.output != None:
    con_config.output_folder = args.output
else:
    con_config.output_folder = "./" + base_name + "_output"
if args.rework:
    print(f"[rework] ignore config strength, tile, openpose")
    con_config.denoising_strength = 0.65
    con_config.controlnet = [
        {"module": "depth_midas", "model": "control_v11f1p_sd15_depth_fp16 [4b72d323]", "pixel_perfect": True, "control_mode": 0, "weight": 0.4},
        {"module": "dw_openpose_full", "model": "control_v11p_sd15_openpose_fp16 [73c2b67d]", "control_mode": 2, "pixel_perfect": True, "weight": 1.5},
    ]
else:
    con_config.denoising_strength = args.strength
    if args.tile:
        con_config.controlnet.append({"module": "none", "model": "control_v11f1e_sd15_tile_fp16 [3b860298]", "weight": 1, "control_mode": 0, "pixel_perfect": True})
    if args.openpose:
        con_config.controlnet.append({"module": "dw_openpose_full", "model": "control_v11p_sd15_openpose_fp16 [73c2b67d]", "control_mode": 2, "pixel_perfect": True, "weight": 1})

if args.prompt != "":
    con_config.base_prompt = args.prompt + "," + con_config.base_prompt

print(f"{args=}")

con_config.temporalnet = "v2"
con_config.temporalnet_weight = 0.25

if config["lora_map"]:
    lora_prompt = ""
    for l in config["lora_map"]:
        base_name = os.path.splitext(os.path.basename(l))[0]
        weight = config["lora_map"][l]
        lora_prompt = f"{lora_prompt} <lora:{base_name}:{weight}>"
    con_config.base_prompt2 = lora_prompt

print(con_config.base_prompt)
print(con_config.base_prompt2)

if not args.skip:
    enhancer.run(
        con_config,
        project_folder=configpath,
        overwrite=False,
        reverse=False,
        resume_frame=0,
        start_frame=1,
        end_frame=0,
        rework_mode="",
    )

video_util.combine(os.path.join(configpath, con_config.output_folder), os.path.join(configpath, "output.mp4"), "", 12, "%03d.png", 0)
video_util.combine(os.path.join(configpath, con_config.output_folder), os.path.join(configpath, "output.mp4"), "", 12, "%04d.png", 0)
video_util.combine(os.path.join(configpath, con_config.output_folder), os.path.join(configpath, "output.mp4"), "", 12, "%08d.png", 0)
