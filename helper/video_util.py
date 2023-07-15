import subprocess
import os


def extract(video_path, output_dir, fps=15, format="%07d.png"):
    os.makedirs(output_dir, exist_ok=True)
    subprocess.call(f'ffmpeg -i "{video_path}" -vf fps={fps} "{output_dir}/{format}"', shell=True)


def combine(images_dir, output_path, sound_video_path="", fps=15, format="%07d.png", start_number=1, crf=17):
    if sound_video_path:
        subprocess.call(
            f'ffmpeg -y -r {fps} -start_number {start_number} -i "{images_dir}/{format}" c:v libx264 -pix_fmt yuv420p -crf {crf} "{output_path}"',
            shell=True,
        )
    else:
        subprocess.call(
            f'ffmpeg -y -r {fps} -start_number {start_number} -i "{images_dir}/{format}" -i {sound_video_path} -c:a copy -c:v libx264 -pix_fmt yuv420p -crf {crf} -map 0:v -map 1:a "{output_path}"',
            shell=True,
        )
