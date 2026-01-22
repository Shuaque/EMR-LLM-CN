# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import cv2
import random
import numpy as np

def load_video(path):
    """
    Load video frames from path, convert to grayscale.
    Return: numpy array of shape [T, H, W]
    """
    for i in range(3):  # retry up to 3 times
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            cap.release()
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")


class Compose(object):
    """Compose several preprocess transforms together."""
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        return f"Compose({self.preprocess})"


class Normalize(object):
    """Normalize frames with mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        return (frames - self.mean) / self.std

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class CenterCrop(object):
    """Crop video frames at the center."""
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw) / 2.0))
        delta_h = int(round((h - th) / 2.0))
        return frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]


class RandomCrop(object):
    """Crop video frames randomly."""
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        return frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]

    def __repr__(self):
        return f"RandomCrop(size={self.size})"


class HorizontalFlip(object):
    """Flip video frames horizontally with given probability."""
    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for i in range(t):
                frames[i] = cv2.flip(frames[i], 1)
        return frames
