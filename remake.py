import os
import json
import glob
from PIL import Image
import argparse
import helper.enhancer as enhancer
import helper.generate as generate
import helper.dynamic_zoom as dynamic_zoom
from helper.config import Config, PatchConfig
from helper.util import get_image_paths


def get_args():
    parser = argparse.ArgumentParser()
    # run
    parser.add_argument("remakepath", help="remake path")
    parser.add_argument("--prompt", dest="prompt", default="")
    parser.add_argument("--width", dest="width", default=512)
    parser.add_argument("--height", dest="height", default=768)
    parser.add_argument("--output", dest="output", default=None)

    return parser.parse_args()

def get_geninfo(image):
    im = Image.open(input_image)
    im.load()
    info = {}
    if "parameters" in im.info:
        prompt = ""
        neg_prompt = ""
        extra = ""
        start_neg_prompt = False
        parameters = im.info["parameters"]
        for line in parameters.split("\n"):
            if start_neg_prompt == True:
                if line.startswith("Steps:"):
                    extra = line
                    break
                else:
                    neg_prompt = neg_prompt + line
            else:
                if line.startswith("Negative prompt:"):
                    start_neg_prompt = True
                    neg_prompt = line[len("Negative prompt:") :]
                else:
                    prompt = prompt + line

        if prompt != "":
            info["prompt"] = prompt
        if neg_prompt != "":
            info["neg_prompt"] = neg_prompt
        if extra != "":
            # print(f"[extra] {extra}")
            setting = dict()
            x = extra.split(",")
            for i in x:
                try:
                    key = i[: i.index(":")]
                    val = i[i.index(":") + 1 :]
                    setting[str(key).strip()] = str(val).strip()
                except Exception as e:
                    print("error", e)

            info.update(setting)
        return info


args = get_args()

configpath = os.path.abspath(os.path.join(args.remakepath, os.pardir))

con_config = Config()
con_config.generate_input_folder = args.remakepath
con_config.generate_output_folder = args.remakepath+"_output"
con_config.base_prompt = args.prompt
con_config.seed_mode = "fixed"

input_images_path_list = get_image_paths(args.remakepath)

frame_schedule = {}
for frame_index, input_image in enumerate(input_images_path_list):
    frame_number = frame_index+1

    info = get_geninfo(input_image)

    setting = {}
    setting["base_prompt"] = info["prompt"]
    setting["neg_prompt"] = info["neg_prompt"]
    setting["seed"] = info["Seed"]
    setting["generate_sampler_name"] = info["Sampler"] if "Sampler" in info else "Euler a"
    setting["generate_sampler_step"] = info["Steps"] if "Steps" in info else 25

    frame_schedule[frame_number] = setting

con_config.frame_schedule = frame_schedule
con_config.generate_width = args.width
con_config.generate_height = args.height

# con_config.sampler_name = sampler_name
# con_config.sampler_step = config["steps"]

# con_config.frame_width = image.width * 2
# con_config.frame_height = image.height * 2
# con_config.input_folder = "./" + base_name
# if args.output != None:
#     con_config.output_folder = args.output

# if args.prompt != "":
#     con_config.base_prompt = args.prompt + "," + con_config.base_prompt

print(f"{configpath=}")
print(con_config)

con_config.temporalnet = ""

generate.run(
    con_config,
    project_folder=configpath,
    overwrite=True,
    resume_frame=0,
    start_frame=1,
    end_frame=0,
)
