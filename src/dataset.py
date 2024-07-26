import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CaptchaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        self.labels = labels  # 这里的 labels 是字符串列表，例如 ['12345', '67890', ...]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        label_tensor = torch.tensor([int(char) for char in label], dtype=torch.long)
        return img, label_tensor