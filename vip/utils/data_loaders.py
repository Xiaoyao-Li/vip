# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import glob
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random


def get_ind(vid, index, ds="ego4d"):
    if ds in ["ego4d"]:
        return torchvision.io.read_image(f"{vid}{index:06}.jpg")
    elif ds in ['epic-kitchen']:
        return torchvision.io.read_image(f"{vid}/frame_{index:010d}.jpg")
    else:
        try:
            return torchvision.io.read_image(f"{vid}/{index}.jpg")
        except: 
            return torchvision.io.read_image(f"{vid}/{index}.png")

## Data Loader for VIP
class VIPBuffer(IterableDataset):
    def __init__(self, datasource='ego4d', datapath=None, num_workers=4, 
                 doaug = "none", agentago=False
                 ):
        self._num_workers = max(1, num_workers)
        self.datasource = datasource
        self.datapath = datapath
        assert(datapath is not None)
        self.doaug = doaug
        
        # Augmentations
        self.preprocess = torch.nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224)
                )
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale = (0.2, 1.0)),
            )
        else:
            self.aug = lambda a : a

        # Load Data
        if "ego4d" == self.datasource:
            print("Ego4D")
            self.manifest = pd.read_csv(f"{self.datapath}/manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)
        elif self.datasource in ['epic-kitchen']:
            print("Load Epic-Kitchen Dataset")
            self.manifest = pd.read_csv(f"{self.datapath}/EPIC100_annotations.csv")
            print(self.manifest)
            print(f'keys: {self.manifest.columns}')
            self.epiclen = len(self.manifest)
            self.agentago = agentago

    def _sample(self):
        # Sample a video from datasource
        if self.datasource == 'ego4d':
            vidid = np.random.randint(0, self.ego4dlen)
            m = self.manifest.iloc[vidid]
            vidlen = m["len"]
            vid = m["path"]
        elif self.datasource == 'epic-kitchen':
            vidid = np.random.randint(0, self.epiclen)
            m = self.manifest.iloc[vidid]
            part_id = m['participant_id']
            video_id = m['video_id']
            if self.agentago:
                vid = f"{self.datapath}/{part_id}/agentago_frames/{video_id}"
            else:
                vid = f"{self.datapath}/{part_id}/rgb_frames/{video_id}"
            start_frame = m['start_frame']
            stop_frame = m['stop_frame']
        else: 
            video_paths = glob.glob(f"{self.datapath}/[0-9]*")
            num_vid = len(video_paths)

            video_id = np.random.randint(0, int(num_vid)) 
            vid = f"{video_paths[video_id]}"

            # Video frames must be .png or .jpg
            vidlen = len(glob.glob(f'{vid}/*.png'))
            if vidlen == 0:
                vidlen = len(glob.glob(f'{vid}/*.jpg'))

        # Sample (o_t, o_k, o_k+1, o_T) for VIP training
        if self.datasource in ['epic-kitchen']:
            start_ind = np.random.randint(start_frame, stop_frame-2)
            end_ind = np.random.randint(start_ind+1, stop_frame)
            s0_ind_vip = np.random.randint(start_ind, end_ind)
            s1_ind_vip = min(s0_ind_vip+1, end_ind)
            reward = float(s0_ind_vip == end_ind) - 1
        else:
            start_ind = np.random.randint(0, vidlen-2)  
            end_ind = np.random.randint(start_ind+1, vidlen)
            s0_ind_vip = np.random.randint(start_ind, end_ind)
            s1_ind_vip = min(s0_ind_vip+1, end_ind)
            # Self-supervised reward (this is always -1)
            reward = float(s0_ind_vip == end_ind) - 1

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, self.datasource) 
            img = get_ind(vid, end_ind, self.datasource)
            imts0_vip = get_ind(vid, s0_ind_vip, self.datasource)
            imts1_vip = get_ind(vid, s1_ind_vip, self.datasource)
            
            allims = torch.stack([im0, img, imts0_vip, imts1_vip], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0_vip = allims_aug[2]
            imts1_vip = allims_aug[3]

        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, self.datasource) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, self.datasource) / 255.0) * 255.0
            imts0_vip = self.aug(get_ind(vid, s0_ind_vip, self.datasource) / 255.0) * 255.0
            imts1_vip = self.aug(get_ind(vid, s1_ind_vip, self.datasource) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0_vip, imts1_vip])
        im = self.preprocess(im)
        return (im, reward)

    def __iter__(self):
        while True:
            yield self._sample()