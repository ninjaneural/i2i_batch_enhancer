import os
import glob
from PIL import Image
from helper.config import Config


def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def run(config: Config, project_folder: str, padding_end: int):
    if project_folder:
        output_folder = os.path.normpath(os.path.join(project_folder, config.output_folder))
    else:
        output_folder = config.output_folder

    output_images_path_list = get_image_paths(output_folder)

    if padding_end > 0:
        last_index = len(output_images_path_list)
        last_filename = os.path.basename(output_images_path_list[0])
        filename_length = len(os.path.splitext(last_filename)[0])

        last_image = Image.open(output_images_path_list[last_index - 1])

        for i in range(last_index, last_index + 15):
            print(f"padding_end frame {i+1}")
            filename = str(i + 1).zfill(filename_length)
            last_image.save(os.path.join(output_folder, filename + ".png"))
