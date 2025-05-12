import pandas as pd
import os 
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class GRSDataset(Dataset):
    def __init__(self, path, path_osats, num_frames= 1000, transforms=None):     
        osats = pd.read_excel(path_osats, engine="openpyxl")

        video_labels = (
            osats.groupby("VIDEO")["GLOBA_RATING_SCORE"]
            .mean()
            .apply(self.grs_to_class)
            .to_dict()
        )

        self.videos = self.collect_video_frame_paths(path)
        self.labels = video_labels
        self.num_frames = num_frames
        self.transforms = transforms

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        video_name = list(self.videos.keys())[idx]
        frame_paths = self.videos[video_name]
        label = self.labels[video_name]

        # only select the first num_frames frames
        if len(frame_paths) >= self.num_frames:
            frame_paths = frame_paths[:self.num_frames]
        else: # else repeat the last frame
            frame_paths += [frame_paths[-1]] * (self.num_frames - len(frame_paths))
        
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            else:
                img = transforms.ToTensor()(img)
            frames.append(img)

        # Stack the frames into a tensor
        video_tensor = torch.stack(frames)  # [T, C, H, W]

        return video_tensor, label

    @staticmethod
    def grs_to_class(grs):
        if 8 <= grs <= 15:
            return 0  # novice
        elif 16 <= grs <= 23:
            return 1  # intermediate
        elif 24 <= grs <= 31:
            return 2  # proeficient
        elif 32 <= grs <= 40:
            return 3  # specialist
        else:
            return -1  # GRS not in range
        
    @staticmethod
    def collect_video_frame_paths(data_path):
        '''
        Creates a dictionary with the video name as key and the list of frame paths as value.
        '''
        video_frames_dict = {}
        
        for package in os.listdir(data_path):
            package_path = os.path.join(data_path, package)
            
            if os.path.isdir(package_path):
                for video in os.listdir(package_path):
                    video_path = os.path.join(package_path, video)
                    
                    if os.path.isdir(video_path):
                        frame_paths = [
                            os.path.join(video_path, filename)
                            for filename in os.listdir(video_path)
                            if filename.endswith('.jpg')
                        ]
                        frame_paths.sort()
                        video_frames_dict[video] = frame_paths
                        
        return video_frames_dict
