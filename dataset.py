#-*- coding:utf-8 -*-
# +
import os
import re
import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class JPGPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            transform=None
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.transform = transform

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*.jpg')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*.jpg')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            pairs.append((input_file, target_file))
        return pairs

    def read_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((256, 256))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        return img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file)
            if self.transform:
                input_img = self.transform(input_img)
                input_tensors.append(input_img)
        return torch.stack(input_tensors).cuda()
    
    
    def sample_pairs(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        samples = [self.__getitem__(index) for index in indexes]
        input_imgs, target_imgs = zip(*[(sample['input'], sample['target']) for sample in samples])
        return torch.stack(input_imgs), torch.stack(target_imgs)

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file)
        target_img = self.read_image(target_file)

        if self.transform:
            combined_img = np.concatenate((input_img, target_img), axis=2)
            combined_img = self.transform(combined_img)
            input_img, target_img = torch.split(combined_img, 3, dim=0)

        return {'input': input_img, 'target': target_img}
# -


