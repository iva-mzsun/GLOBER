import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from ipdb import set_trace as st

import decord
from decord import VideoReader
from data.utils import center_crop

class VideoDataset(Dataset):
    def __init__(self, root, extract_fps, clip_nframe, keyframe_idx,
                 ids_file=None, id2cap_file=None, fix_prompt=None,
                 shuffle=True, size=None, max_data_num=None):
        self.size = size
        self.root = root
        self.fix_prompt = fix_prompt
        self.extract_fps = extract_fps
        self.clip_nframe = clip_nframe
        self.keyframe_idx = keyframe_idx
        self.interpolation = Image.BICUBIC
        self.clip_duration = clip_nframe / extract_fps

        # load id2cap dict
        if id2cap_file is None:
            self.id2cap = None
        elif id2cap_file.endswith('json'):
            self.id2cap = json.load(open(id2cap_file))
        else:
            raise NotImplementedError
        # load data
        if ids_file is None:
            videos = sorted(os.listdir(root))
        else:
            videos = sorted(json.load(open(ids_file)))
        self.items = videos

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size)
        ])

        if shuffle:
            print("- NOTE: shuffle video items!")
            from random import shuffle
            shuffle(self.items)
        if max_data_num is not None:
            self.items = self.items[:max_data_num]

    def __len__(self):
        return len(self.items)

    # def load_img(self, img_path, if_flip):
    #     image = Image.open(img_path)
    #     if not image.mode == "RGB":
    #         image = image.convert("RGB")
    #     image = center_crop(image)
    #     if self.size is not None:
    #         image = image.resize((self.size, self.size),
    #                              resample=self.interpolation)
    #     if if_flip:
    #         image = F.hflip(image)
    #     image = np.array(image).astype(np.uint8)
    #     image = image.transpose((2, 0, 1))
    #     image = (image / 127.5 - 1.0).astype(np.float32)
    #     return image

    def __skip_sample__(self, idx):
        if idx == len(self.items) - 1:
            return self.__getitem__(0)
        else:
            return self.__getitem__(idx+1)

    def __random_sample__(self):
        idx = np.random.randint(0, len(self.items))
        return self.__getitem__(idx)

    def get_frames(self, curv):
        # read video
        path = os.path.join(self.root, curv)
        container = VideoReader(path)
        fps = container.get_avg_fps()
        nframe = len(container)

        nframe_total = int(fps * self.clip_duration)
        start = random.choice(range(nframe - nframe_total + 1))
        frame_skip = (nframe_total - 1) // self.clip_nframe
        clip_indexes = range(0, nframe_total, frame_skip)
        clip_indexes = clip_indexes[:self.clip_nframe]
        sample_idx = [i + start for i in clip_indexes]
        # print(fps, self.extract_fps, self.clip_duration)
        # print(nframe_total, frame_skip, len(sample_idx), sample_idx)
        frames = container.get_batch(sample_idx).asnumpy()
        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))
        # print(1, frames.shape, torch.min(frames), torch.max(frames))
        frames = frames.to(torch.float32) / 127.5 - 1.0
        frames = self.transform(frames)
        # print(2, frames.shape, torch.min(frames), torch.max(frames))
        return frames

    def __getitem__(self, idx):
        curv = self.items[idx]
        # obtain video caption
        if self.id2cap is None:
            videoname = "vnull"
            prompt = self.fix_prompt
        else:
            videoname = curv
            prompt = self.id2cap[curv]

        # load video frames
        try:
            frames = self.get_frames(curv)
        except Exception as e:
            # print(f"Failed with Exception: {e}")
            return self.__skip_sample__(idx)

        # load video key frames
        key_frames = []
        for i in self.keyframe_idx:
            key_frames.append(frames[i][:, np.newaxis, :, :])
        key_frames = np.concatenate(key_frames, axis=1)

        # random select target frame indexes
        indexes = []
        for tarindex in range(self.clip_nframe):
            preindex = tarindex if tarindex == 0 else tarindex - 1
            indexes.append((tarindex, preindex))
        indexes = sorted(indexes, key=lambda x: x[0])
        tar_indexes = np.array([ind[0] for ind in indexes])
        pre_indexes = np.array([ind[1] for ind in indexes])

        # load target video frames
        tar_frames = []
        for ind in tar_indexes:
            tar_frames.append(frames[ind][:, np.newaxis, :, :])
        tar_indexes = tar_indexes.astype(np.float) / self.clip_nframe
        tar_frames = np.concatenate(tar_frames, axis=1)

        # load previous video frames
        pre_frames = []
        for ind in pre_indexes:
            pre_frames.append(frames[ind][:, np.newaxis, :, :])
        pre_indexes = pre_indexes.astype(np.float) / self.clip_nframe
        pre_frames = np.concatenate(pre_frames, axis=1)

        # full video indexes
        full_indexes = np.arange(self.clip_nframe).astype(np.float) / self.clip_nframe

        return dict({
            'txt': prompt, # str
            'video_name': videoname,
            'tar_frames': tar_frames,
            'frame_index': tar_indexes,
            'pre_frames': pre_frames,
            'preframe_index': pre_indexes,
            'key_frame': key_frames,
            'full_index': full_indexes
        })
