import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from ipdb import set_trace as st

from data.utils import center_crop

class VideoFolderDataset(Dataset):
    def __init__(self, root, num_frames, id2cap_file=None, ids_file=None,
                 allow_flip=True, sort_index=False, shuffle=True,
                 content_frame_idx=(0, 10, 20, 31), full_video_length=32,
                 fix_prompt=None, flip_p=0.5, size=None, max_data_num=None):
        self.size = size
        self.root = root
        self.flip_prob = flip_p
        self.sort_index = sort_index
        self.num_frames = num_frames
        self.allow_filp = allow_flip
        self.fix_prompt = fix_prompt
        self.content_frame_idx = content_frame_idx
        self.full_video_length = full_video_length
        self.interpolation = Image.BICUBIC
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
        self.videos = videos

        if shuffle:
            print("- NOTE: shuffle video items!")
            from random import shuffle
            shuffle(self.videos)
        if max_data_num is not None:
            self.videos = self.videos[:max_data_num]

    def __len__(self):
        return len(self.videos)

    def load_img(self, img_path, if_flip):
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = center_crop(image)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)
        if if_flip:
            image = F.hflip(image)
        image = np.array(image).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def get_frames(self, curv):
        vdir = os.path.join(self.root, curv)
        frames = sorted(os.listdir(vdir))
        frames = [os.path.join(vdir, f) for f in frames]
        return frames

    def __skip_sample__(self, idx):
        if idx == len(self.videos) - 1:
            return self.__getitem__(0)
        else:
            return self.__getitem__(idx+1)

    def __random_sample__(self):
        idx = np.random.randint(0, len(self.videos))
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        curv = self.videos[idx]
        if self.id2cap is None:
            videoname = "vnull"
            prompt = self.fix_prompt
        else:
            videoname = curv
            prompt = self.id2cap[curv]

        # determine if flip
        p = np.random.rand()
        if_flip = p < self.flip_prob \
            if self.allow_filp else False

        # load video frames * point
        frames = self.get_frames(curv)
        if len(frames) < self.full_video_length:
            # print(f"Skip video id: {curv}")
            return self.__random_sample__()
        end = len(frames) - self.full_video_length
        point = np.random.choice(end + 1)

        # load video content frames
        content_frames = []
        for i in self.content_frame_idx:
            cframe = self.load_img(frames[point + i], if_flip)
            content_frames.append(cframe[:, np.newaxis, :, :])
        content_frames = np.concatenate(content_frames, axis=1)

        # random select target frame indexes
        video_length = self.full_video_length
        indexes = []
        for _ in range(self.num_frames):
            tarindex = np.random.randint(0, video_length)
            preindex = tarindex if tarindex == 0 else tarindex - 1
            indexes.append((tarindex, preindex))
        if self.sort_index:
            indexes = sorted(indexes, key=lambda x: x[0])
        tar_indexes = np.array([ind[0] for ind in indexes])
        pre_indexes = np.array([ind[1] for ind in indexes])

        # load target video frames
        tar_frames = []
        for ind in tar_indexes:
            tar_frame = self.load_img(frames[point + ind], if_flip)
            tar_frames.append(tar_frame[:, np.newaxis, :, :])
        tar_indexes = tar_indexes.astype(np.float) / video_length
        tar_frames = np.concatenate(tar_frames, axis=1)

        # load previous video frames
        pre_frames = []
        for ind in pre_indexes:
            pre_frame = self.load_img(frames[point + ind], if_flip)
            pre_frames.append(pre_frame[:, np.newaxis, :, :])
        pre_indexes = pre_indexes.astype(np.float) / video_length
        pre_frames = np.concatenate(pre_frames, axis=1)

        full_indexes = np.arange(video_length).astype(np.float) / video_length

        return dict({
            'txt': prompt, # str
            'video_name': videoname,
            'tar_frames': tar_frames,
            'frame_index': tar_indexes,
            'pre_frames': pre_frames,
            'preframe_index': pre_indexes,
            'key_frame': content_frames,
            'full_index': full_indexes
        })
