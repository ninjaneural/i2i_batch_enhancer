import os
import json
import glob
from PIL import Image
import argparse
import helper.enhancer as enhancer
import helper.enhance_face as enhance_face
import helper.enhance_face_fix as enhance_face_fix
import helper.enhance_zoom as enhance_zoom
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
    parser.add_argument("configfile", help="or video file")
    parser.add_argument("--resume-frame", dest="resume_frame", default=0)
    parser.add_argument("--start-frame", dest="start_frame", default=1)
    parser.add_argument("--end-frame", dest="end_frame", default=0)
    parser.add_argument("--overwrite", dest="overwrite", default=False)
    parser.add_argument("--padding-end", dest="padding_end", default=0)
    parser.add_argument("--rework-mode", dest="rework_mode", default="")
    parser.add_argument("--patchfile", dest="patchfile", default="")
    parser.add_argument("--samplerun", dest="samplerun", default=None)
    parser.add_argument("--loratest", dest="loratest", default=None)
    parser.add_argument("--generate", dest="generate", default=False)
    parser.add_argument("--dynamic-zoom", dest="dynamic_zoom", default=False)
    parser.add_argument("--reverse", dest="reverse", default=False)
    parser.add_argument("--face-enhance", dest="face_enhance", default=False)
    parser.add_argument("--face-enhance-fix", dest="face_enhance_fix", default=False)
    parser.add_argument("--zoom-enhance", dest="zoom_enhance", default=False)

    # video
    parser.add_argument("--fps", dest="fps", default=15)
    parser.add_argument("--sound-file", dest="sound_file", default=None)

    return parser.parse_args()

def run():
    args = get_args()

    configpath = os.path.dirname(args.configfile)

    if os.path.splitext(args.configfile)[1] == ".mp4" or os.path.splitext(args.configfile)[1] == ".webm":
        output_path = os.path.normpath(os.path.join(configpath, os.path.splitext(args.configfile)[0]))
        print(f"# video extract mode")
        print(f"video {args.configfile}")
        print(f"fps {args.fps}")
        print(f"output_path {output_path}")
        video_util.extract(args.configfile, output_path, args.fps)
        return

    if os.path.isdir(args.configfile):
        video_util.combine(args.configfile, os.path.normpath(os.path.join(args.configfile, "../output.mp4")), args.sound_file, args.fps)
        return

    with open(args.configfile, "r") as json_read:
        config = json.load(json_read)

    # patch cover mode
    if args.patchfile:
        with open(os.path.join(configpath, args.patchfile), "r") as json_read:
            patchconfig = json.load(json_read)
        patch_cover.run(Config(**config), project_folder=configpath, patchconfig=PatchConfig(**patchconfig))
    # padding end mode
    elif int(args.padding_end) > 0:
        padding_end.run(Config(**config), project_folder=configpath, padding_end=int(args.padding_end))
    # samplerun
    elif args.samplerun != None:
        samplerun.run(Config(**config), project_folder=configpath, samplerun=args.samplerun)
    # loratest
    elif args.loratest:
        loratest.run(Config(**config), project_folder=configpath)
    # dynamic zoom
    elif args.dynamic_zoom:
        dynamic_zoom.run(Config(**config), project_folder=configpath)
    # face enhance
    elif args.face_enhance:
        enhance_face.run(
            Config(**config),
            project_folder=configpath,
            overwrite=bool(args.overwrite),
            reverse=bool(args.reverse),
            resume_frame=int(args.resume_frame),
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
        )
    elif args.face_enhance_fix:
        enhance_face_fix.run(
            Config(**config),
            project_folder=configpath,
            overwrite=bool(args.overwrite),
            reverse=bool(args.reverse),
            resume_frame=int(args.resume_frame),
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
        )
    # zoom enhance
    elif args.zoom_enhance:
        enhance_zoom.run(
            Config(**config),
            project_folder=configpath,
            overwrite=bool(args.overwrite),
            reverse=bool(args.reverse),
            resume_frame=int(args.resume_frame),
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
        )
    # generate
    elif args.generate:
        generate.run(
            Config(**config),
            project_folder=configpath,
            overwrite=bool(args.overwrite),
            resume_frame=int(args.resume_frame),
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
        )
    # run
    else:
        enhancer.run(
            Config(**config),
            project_folder=configpath,
            overwrite=bool(args.overwrite),
            reverse=bool(args.reverse),
            resume_frame=int(args.resume_frame),
            start_frame=int(args.start_frame),
            end_frame=int(args.end_frame),
            rework_mode=str(args.rework_mode),
        )


if __name__=='__main__':
    run()
