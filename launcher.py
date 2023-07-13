import json
import sympy
import argparse
import helper.enhancer as enhancer
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile")
    parser.add_argument("--resume-index", dest="resume_index", default=0)
    return parser.parse_args()


args = get_args()

configpath = os.path.dirname(args.configfile)

with open(args.configfile, "r") as json_read:
    config = json.load(json_read)

# print(config["controlnet"])
# print(config["use_interrogate"])
# print(sympy.sympify(config["zoom_area_limit"]))
# print(sympy.sympify(config["zoom_max_resolusion"]))

enhancer.run(**config, project_path=configpath, resume_index=args.resume_index)
