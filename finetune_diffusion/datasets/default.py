import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
import numpy as np
import pandas as pd
import torch


class DefaultDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        df = pd.read_csv(os.path.join(data_dir, 'captions.csv'), header=None)
        self.filepath_caption_map = {os.path.join(data_dir, 'images', row[0]): row[1].replace('"', '') for _, row in df.iterrows()}

    def __len__(self):
        return len(self.filepath_caption_map)
    
    def __getitem__(self, idx):
        example = {}

        filepath = list(self.filepath_caption_map.keys())[idx]
        caption = self.filepath_caption_map[filepath]

        image = Image.open(filepath)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize((512, 512), resample=PIL.Image.Resampling.BICUBIC)
        image = transforms.RandomHorizontalFlip(p=0.5)(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)

        example["input_ids"] = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return example
