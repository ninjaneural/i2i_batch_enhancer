import os
import json
import argparse
import helper.enhancer as enhancer
import helper.padding_end as padding_end
import helper.video_util as video_util
import helper.patch_cover as patch_cover
from helper.config import Config


def get_args():
    parser = argparse.ArgumentParser()
    # run
    parser.add_argument("configfile", help="or video file")
    parser.add_argument("--resume-frame", dest="resume_frame", default=0)
    parser.add_argument("--end-frame", dest="end_frame", default=0)
    parser.add_argument("--overwrite", dest="overwrite", default=False)
    parser.add_argument("--padding-end", dest="padding_end", default=0)
    parser.add_argument("--rework-mode", dest="rework_mode", default="")
    parser.add_argument("--patchfile", dest="patchfile", default="")

    # video
    parser.add_argument("--fps", dest="fps", default=15)

    return parser.parse_args()


args = get_args()

configpath = os.path.dirname(args.configfile)

if os.path.splitext(args.configfile)[1] == ".mp4":
    output_path = os.path.normpath(os.path.join(configpath, os.path.splitext(args.configfile)[0]))
    print(f"# video extract mode")
    print(f"video {args.configfile}")
    print(f"fps {args.fps}")
    print(f"output_path {output_path}")
    video_util.extract(args.configfile, output_path, args.fps)

else:
    with open(args.configfile, "r") as json_read:
        config = json.load(json_read)

    # patch cover mode
    if args.patchfile:
        with open(os.path.join(configpath, args.patchfile), "r") as json_read:
            patchconfig = json.load(json_read)
        patch_cover.run(Config(**config), project_folder=configpath, patch_name=args.patchfile, patch_rects=patchconfig)
    # padding end mode
    elif args.padding_end > 0:
        padding_end.run(Config(**config), project_folder=configpath, padding_end=int(args.padding_end))
    # run
    else:
        enhancer.run(Config(**config), project_folder=configpath, overwrite=bool(args.overwrite), resume_frame=int(args.resume_frame), end_frame=int(args.end_frame), rework_mode=str(args.rework_mode))
