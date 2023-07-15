import json
import sympy
import argparse
import helper.enhancer as enhancer
import helper.video_util as video_util
import os


def get_args(a=1):
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", help="or video file")
    parser.add_argument("--fps", dest="fps", default=15)
    parser.add_argument("--resume-index", dest="resume_index", default=0)
    b = 1
    exec("a = 2")
    exec("b = 2")
    print(f"a {a} b {b}")

    return parser.parse_args()


args = get_args()

configpath = os.path.dirname(args.configfile)

if os.path.splitext(args.configfile)[1] == ".mp4":
    output_path = os.path.normpath(os.path.join(configpath, "./" + os.path.splitext(args.configfile)[0]))
    print(f"# video extract mode")
    print(f"video {args.configfile}")
    print(f"fps {args.fps}")
    print(f"output_path {output_path}")
    video_util.extract(args.configfile, output_path, args.fps)

else:
    with open(args.configfile, "r") as json_read:
        config = json.load(json_read)

    # enhancer.run(**config, project_folder=configpath, resume_index=args.resume_index, config=config)
