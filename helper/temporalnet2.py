import os
import base64
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
import pickle
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

def make_flow(frameA, frameB, frame_width, frame_height, flow_image_folder, output_filename):
    input_frame_1 = read_image(str(frameA), ImageReadMode.RGB)
    input_frame_2 = read_image(str(frameB), ImageReadMode.RGB)

    img1_batch = torch.stack([input_frame_1])
    img2_batch = torch.stack([input_frame_2])

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    def preprocess(img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[512, 512], antialias=True)
        img2_batch = F.resize(img2_batch, size=[512, 512], antialias=True)
        return transforms(img1_batch, img2_batch)

    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))

    predicted_flow = list_of_flows[-1][0]

    flow_img = flow_to_image(predicted_flow)
    img_arr = np.array(flow_img.permute(1, 2, 0).cpu(), dtype=np.uint8)
    img_arr = cv2.resize(img_arr, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

    flow_image_path = os.path.join(flow_image_folder, output_filename)
    Image.fromarray(img_arr).save(flow_image_path)

    return img_arr


def encode_image(flow_image_array, last_image_array):
    # Concatenating the three images to make a 6-channel image
    six_channel_image = np.dstack((last_image_array, flow_image_array))

    # Serializing the 6-channel image
    serialized_image = pickle.dumps(six_channel_image)

    # Encoding the serialized image
    encoded_image = base64.b64encode(serialized_image).decode("utf-8")

    return encoded_image
