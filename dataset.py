import os
import cv2
import yaml
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

class Letterbox:
    """
    Resize image to a strict square (e.g., 640x640) while preserving the original aspect ratio.
    Pads the shorter side with a gray background (value=114).
    """
    def __init__(self, target=640, color=(114, 114, 114)):
        self.target = target
        self.color  = color

    def __call__(self, image, labels):
        H, W = image.shape[:2]
        t = self.target

        r = min(t / H, t / W)
        new_H = int(round(H * r))
        new_W = int(round(W * r))

        resized = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        pad_h = t - new_H
        pad_w = t - new_W
        top = pad_h // 2
        bot = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        img_lb = cv2.copyMakeBorder(
            resized, top, bot, left, right,
            cv2.BORDER_CONSTANT, value=self.color
        )

        if labels.shape[0] > 0:
            labels = labels.copy().astype(np.float32)
            labels[:, 1] = labels[:, 1] * W          
            labels[:, 2] = labels[:, 2] * H          
            labels[:, 3] = labels[:, 3] * W          
            labels[:, 4] = labels[:, 4] * H          

            labels[:, 1:] = labels[:, 1:] * r

            labels[:, 1] += left
            labels[:, 2] += top

            labels[:, 1] = labels[:, 1] / t          
            labels[:, 2] = labels[:, 2] / t          
            labels[:, 3] = labels[:, 3] / t          
            labels[:, 4] = labels[:, 4] / t          

            labels[:, 1:] = labels[:, 1:].clip(0.0, 1.0)

        return img_lb, labels


def augment_hsv(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv.astype(np.float32))
    h = (h * r[0]) % 180
    s = np.clip(s * r[1], 0, 255)
    v = np.clip(v * r[2], 0, 255)

    hsv_aug = cv2.merge([h, s, v]).astype(np.uint8)
    cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2RGB, dst=image)


def augment_hflip(image, labels):
    image = image[:, ::-1, :].copy()
    if labels.shape[0] > 0:
        labels = labels.copy()
        labels[:, 1] = 1.0 - labels[:, 1]    
    return image, labels


def load_label(label_path):
    path = Path(label_path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    labels = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5: 
                labels.append([float(x) for x in parts])

    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(labels, dtype=np.float32)


class PolyDataset(Dataset):
    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, yaml_path, split='train', img_size=640, augment=False):
        self.img_size = img_size
        self.augment  = augment
        self.letterbox = Letterbox(target=img_size)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        self.nc    = cfg.get('nc', 1)
        self.names = cfg.get('names', ['polyp'])
        base = Path(yaml_path).parent
        
        split_key = split
        if split_key not in cfg and split == 'valid' and 'val' in cfg: split_key = 'val'
        if split_key not in cfg and split == 'val' and 'valid' in cfg: split_key = 'valid'

        split_path = Path(cfg[split_key])
        if not split_path.is_absolute():
            if not split_path.exists():
                split_path = base / split_path

        img_dir = split_path / 'images'
        if not img_dir.exists():
            img_dir = split_path

        self.img_paths = sorted([
            p for p in img_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])

        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

        self.lbl_paths = []
        for img_p in self.img_paths:
            lbl_p = Path(str(img_p).replace(
                os.sep + 'images' + os.sep,
                os.sep + 'labels' + os.sep
            )).with_suffix('.txt')
            self.lbl_paths.append(lbl_p)

        print(f"[PolyDataset] split={split}  images={len(self.img_paths)}  augment={augment}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]))
        if img is None:
            raise IOError(f"Could not read image: {self.img_paths[idx]}")

        labels = load_label(self.lbl_paths[idx])   
        img, labels = self.letterbox(img, labels)

        if self.augment:
            augment_hsv(img)
            if random.random() < 0.5:
                img, labels = augment_hflip(img, labels)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous() 
        labels = torch.from_numpy(labels)   

        return img, labels


def collate_fn(batch):
    imgs, label_list = zip(*batch)
    imgs = torch.stack(imgs, dim=0)           

    max_gt = max(lbl.shape[0] for lbl in label_list)
    max_gt = max(max_gt, 1)                   

    B = len(label_list)
    padded = torch.zeros(B, max_gt, 5)
    for i, lbl in enumerate(label_list):
        if lbl.shape[0] > 0:
            padded[i, :lbl.shape[0]] = lbl

    return imgs, padded                 


def build_dataloaders(yaml_path, img_size=640, batch_size=16, num_workers=2):
    train_ds = PolyDataset(yaml_path, split='train', img_size=img_size, augment=True)
    val_ds   = PolyDataset(yaml_path, split='valid', img_size=img_size, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

