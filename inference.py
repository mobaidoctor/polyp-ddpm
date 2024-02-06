#-*- coding:utf-8 -*-
from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.unet import create_model
from torchvision import utils
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
import argparse
import glob
import cv2
import os
import numpy as np
from imgaug import augmenters as iaa
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import uuid
from torchvision.transforms import Compose, Lambda
from PIL import Image
import numpy as np
import torch
import os 
import glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# +
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="dataset/test/masks")
parser.add_argument('-e', '--exportfolder', type=str, default='exports')
parser.add_argument('-w', '--weightfile', type=str, default='models/model.pt')
parser.add_argument('-d', '--device', type=str, default='cuda')
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--num_samples', type=int, default=4)

args = parser.parse_args()
# -

inputfolder = args.inputfolder
exportfolder = args.exportfolder
input_size = args.input_size
batchsize = args.batchsize
weightfile = args.weightfile
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_samples = args.num_samples
in_channels = 6
out_channels = 3
device = "cuda"

mask_list = sorted(glob.glob(f"{inputfolder}/*.jpg"))
print("Total input masks: ", len(mask_list))

input_transform = Compose([
    ToPILImage(),
    Resize(input_size),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1)
])

model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

# +
weight = torch.load(weightfile)
diffusion.load_state_dict(weight['ema'])
print("Model Loaded!")

img_dir = exportfolder + "/image"   
msk_dir = exportfolder + "/mask"   
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)
# -

for k, inputfile in enumerate(mask_list):
    left = len(mask_list) - (k+1)
    print("MASKS LEFT: ", left)
    batches = num_to_groups(num_samples, batchsize)
    img = Image.open(inputfile)
    img = img.resize((256, 256))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)
    input_tensor = input_transform(img)
    input_tensor = input_tensor.unsqueeze(0)
    msk_name = inputfile.split('/')[-1]
    
    steps = len(batches)
    sample_count = 0
    
    print(f"All Step: {steps}")
    counter = 0
    
    for i, bsize in enumerate(batches):
        print(f"Step [{i+1}/{steps}]")
        condition_tensors, counted_samples = [], []
        for b in range(bsize):
            condition_tensors.append(input_tensor)
            counted_samples.append(sample_count)
            sample_count += 1

        condition_tensors = torch.cat(condition_tensors, 0).cuda()
        imgs_list = list(map(lambda n: diffusion.sample(batch_size=n, condition_tensors=condition_tensors), [bsize]))

        # Iterate over each batch and each image in the batch
        for batch_idx, imgs in enumerate(imgs_list):
            imgs = (imgs + 1) * 0.5  # Normalize the images
            
            for img_idx, img in enumerate(imgs):
                counter = counter + 1
                # Generate a unique filename for each image
                filename = os.path.join(img_dir, f'{counter}-{msk_name}')
                utils.save_image(img, filename)
                # Generate a unique filename for each image
                filename = os.path.join(msk_dir, f'{counter}-{msk_name}')
                utils.save_image(condition_tensors[0], filename)
        print("Done!")
