import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

def center_crop(im): # im: PIL.Image
    width, height = im.size   # Get dimensions
    new_width = min(width, height)
    new_height = min(width, height)
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

class CustomDataset_1to1(Dataset):
    def __init__(self, root, flip_p=0.5, size=None, max_data_num=None):
        self.size = size
        self.flip_prob = flip_p
        self.interpolation = Image.BICUBIC
        # load data
        self.videos = sorted(os.listdir(root))
        if max_data_num is not None:
            self.videos = self.videos[:max_data_num]
        self.vid2img = dict({})
        self.vidpoint = dict({})
        for v in self.videos:
            self.vidpoint[v] = 0
            vpath = os.path.join(root, v)
            frames = [os.path.join(vpath, f)
                      for f in os.listdir(vpath)]
            self.vid2img[v] = sorted(frames)

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

    def __getitem__(self, idx):
        curv = self.videos[idx]
        prompt = curv.split('_')[1]
        # determine if flip
        p = np.random.rand()
        if_flip = p < self.flip_prob
        # load video frames
        frames = self.vid2img[curv]
        point = self.vidpoint[curv]
        src_frame = self.load_img(frames[point], if_flip)
        tar_frame = self.load_img(frames[point + 1], if_flip)
        # update point
        if point + 1 >= len(frames):
            point = 0
        self.vidpoint[curv] = point + 1

        return dict({
            'txt': prompt,
            'video_name': curv,
            'src_frame': src_frame,
            'tar_frame': tar_frame,
        })

# ================================================
class VideoFolderDataset_AE(Dataset):
    def __init__(self, root, start_zero_p=0.1,
                 content_frame_idx=(0, 15), full_video_length=16,
                 flip_p=0.5, size=None, max_data_num=None):
        self.size = size
        self.flip_prob = flip_p
        self.start_zero_p = start_zero_p # prob of choosing the first video frame
        self.content_frame_idx = content_frame_idx
        self.full_video_length = full_video_length
        self.interpolation = Image.BICUBIC
        # load data
        self.videos = sorted(os.listdir(root))
        if max_data_num is not None:
            self.videos = self.videos[:max_data_num]
        self.vid2img = dict({})
        self.vidpoint = dict({})
        for v in self.videos:
            self.vidpoint[v] = 0
            vpath = os.path.join(root, v)
            frames = [os.path.join(vpath, f)
                      for f in os.listdir(vpath)]
            self.vid2img[v] = sorted(frames)

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

    def update_point(self, curv):
        cpoint = self.vidpoint[curv]
        self.vidpoint[curv] = cpoint + 1
        if cpoint + self.full_video_length >= len(self.vid2img[curv]):
            self.vidpoint[curv] = 0

    def __getitem__(self, idx):
        curv = self.videos[idx]
        prompt = curv.split('_')[1]
        # determine if flip
        p = np.random.rand()
        if_flip = p < self.flip_prob
        # load video frames
        frames = self.vid2img[curv]
        point = self.vidpoint[curv]
        self.update_point(curv)
        # load video content frames
        content_frames = []
        for i in self.content_frame_idx:
            content_frames.append(self.load_img(frames[point + i], if_flip)[:, np.newaxis, :, :])
        content_frames = np.concatenate(content_frames, axis=1)
        # random select a target frame
        p = np.random.rand()
        tar_frame_index = 0 if p < self.start_zero_p else \
            np.random.randint(1, self.full_video_length)
        tar_frame = self.load_img(frames[point + tar_frame_index], if_flip)
        if tar_frame_index == 0:
            src_frame = np.zeros_like(tar_frame)
        else:
            src_frame = self.load_img(frames[point + tar_frame_index - 1], if_flip)
        tar_frame_index /= self.full_video_length

        return dict({
            'txt': prompt, # str
            'video_name': curv, # str
            'src_frame': src_frame, # [c, h, w]
            'tar_frame': tar_frame, # [c, h, w]
            'full_frame': content_frames, # [c, t, h, w]
            'frame_index': tar_frame_index, # float, (0,1)
        })

class VideoFolderDataset_AEwT(Dataset):
    def __init__(self, root, num_frames, allow_flip=True, allow_point=True, sort_index=False,
                 content_frame_idx=(0, 10, 20, 31), full_video_length=32,
                 flip_p=0.5, size=None, max_data_num=None):
        self.size = size
        self.flip_prob = flip_p
        self.sort_index = sort_index
        self.num_frames = num_frames
        self.allow_filp = allow_flip
        self.allow_point = allow_point
        self.content_frame_idx = content_frame_idx
        self.full_video_length = full_video_length
        self.interpolation = Image.BICUBIC
        # load data
        self.videos = sorted(os.listdir(root))
        if max_data_num is not None:
            from random import shuffle
            shuffle(self.videos)
            self.videos = self.videos[:max_data_num]
        self.vid2img = dict({})
        self.vidpoint = dict({})
        for v in self.videos:
            self.vidpoint[v] = 0
            vpath = os.path.join(root, v)
            frames = [os.path.join(vpath, f)
                      for f in os.listdir(vpath)]
            self.vid2img[v] = sorted(frames)

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

    def update_point(self, curv):
        if self.allow_point is False:
            return
        cpoint = self.vidpoint[curv]
        self.vidpoint[curv] = cpoint + 1
        if cpoint + self.full_video_length >= len(self.vid2img[curv]):
            self.vidpoint[curv] = 0

    def __getitem__(self, idx):
        curv = self.videos[idx]
        prompt = curv.split('_')[1]
        # determine if flip
        p = np.random.rand()
        if_flip = p < self.flip_prob \
            if self.allow_filp else False
        # load video frames * point
        frames = self.vid2img[curv]
        point = self.vidpoint[curv]
        self.update_point(curv)
        # load video content frames
        content_frames = []
        for i in self.content_frame_idx:
            content_frames.append(self.load_img(frames[point + i], if_flip)[:, np.newaxis, :, :])
        content_frames = np.concatenate(content_frames, axis=1)
        # random select target frame indexes
        tar_indexes = np.random.randint(0, self.full_video_length, self.num_frames)
        if self.sort_index:
            tar_indexes = np.sort(tar_indexes)
        # load target video frames
        tar_frames = []
        for ind in tar_indexes:
            tar_frame = self.load_img(frames[point + ind], if_flip)
            tar_frames.append(tar_frame[:, np.newaxis, :, :])

        tar_indexes = tar_indexes.astype(np.float) / self.full_video_length
        # tar_indexes =  self.full_video_length
        tar_frames = np.concatenate(tar_frames, axis=1)

        return dict({
            'txt': prompt, # str
            'video_name': curv, # str
            'tar_frames': tar_frames, # [c, t, h, w]
            'full_frame': content_frames, # [c, t, h, w]
            'frame_index': tar_indexes, # [t]
        })

class VideoFolderDataset_AEwTwIDS(Dataset):
    def __init__(self, root, num_frames, ids_json,
                 allow_flip=True, allow_point=True, sort_index=False,
                 content_frame_idx=(0, 10, 20, 31), full_video_length=32,
                 flip_p=0.5, size=None, max_data_num=None):
        self.size = size
        self.flip_prob = flip_p
        self.sort_index = sort_index
        self.num_frames = num_frames
        self.allow_flip = allow_flip
        self.allow_point = allow_point
        self.content_frame_idx = content_frame_idx
        self.full_video_length = full_video_length
        self.interpolation = Image.BICUBIC
        # load video ids
        data = json.load(open(ids_json, 'r'))
        self.videos = data.keys()
        if max_data_num is not None:
            from random import shuffle
            shuffle(self.videos)
            self.videos = self.videos[:max_data_num]
        # obtain video frames & points
        self.vid2img = dict({})
        self.vidpoint = dict({})
        for v in self.videos:
            self.vidpoint[v] = 0
            vpath = os.path.join(root, v)
            frames = [os.path.join(vpath, f)
                      for f in os.listdir(vpath)]
            self.vid2img[v] = sorted(frames)

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

    def update_point(self, curv):
        cpoint = self.vidpoint[curv]
        self.vidpoint[curv] = cpoint + 1
        if cpoint + self.full_video_length >= len(self.vid2img[curv]):
            self.vidpoint[curv] = 0

    def __getitem__(self, idx):
        curv = self.videos[idx]
        prompt = curv.split('_')[1]
        # determine if flip
        if self.allow_flip:
            p = np.random.rand()
            if_flip = p < self.flip_prob
        else:
            if_flip = False
        # load video frames * point
        frames = self.vid2img[curv]
        if self.allow_point:
            point = self.vidpoint[curv]
            self.update_point(curv)
        else:
            point = 0
        # load video content frames
        content_frames = []
        for i in self.content_frame_idx:
            content_frames.append(self.load_img(frames[point + i], if_flip)[:, np.newaxis, :, :])
        content_frames = np.concatenate(content_frames, axis=1)
        # random select target frame indexes
        tar_indexes = np.random.randint(0, self.full_video_length, self.num_frames)
        if self.sort_index:
            tar_indexes = np.sort(tar_indexes)
        # load target video frames
        tar_frames = []
        for ind in tar_indexes:
            tar_frame = self.load_img(frames[point + ind], if_flip)
            tar_frames.append(tar_frame[:, np.newaxis, :, :])

        tar_indexes = tar_indexes.astype(np.float) / self.full_video_length
        tar_frames = np.concatenate(tar_frames, axis=1)

        return dict({
            'txt': prompt, # str
            'video_name': curv, # str
            'tar_frames': tar_frames, # [c, t, h, w]
            'full_frame': content_frames, # [c, t, h, w]
            'frame_index': tar_indexes, # [t]
        })

