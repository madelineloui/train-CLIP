import glob
import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import pandas as pd
from util import get_meta_from_json, create_fmow_caption
import pytorch_lightning as pl
from transformers import CLIPTokenizer

# TODO: should not be hardcoded?
tokenizer = CLIPTokenizer.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")

class FmowDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, csv_file, caption_type, batch_size=32, num_workers=4, val_split=0.2):
        super().__init__()
        self.tokenizer = tokenizer
        self.csv_file = csv_file
        self.caption_type = caption_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        # Load the full dataset
        dataset = FmowDataset(csv_path=self.csv_file, tokenizer=self.tokenizer, caption_type=self.caption_type)
        
        # Split the dataset into training and validation sets
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=custom_collate_fn, 
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=custom_collate_fn, 
            num_workers=self.num_workers, 
            pin_memory=True
        )


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """

    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class FmowDataset(SatelliteDataset):

    def __init__(self, csv_path, tokenizer, caption_type=0, transform=None):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        
        self.tokenizer = tokenizer
        
        self.mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        self.std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        
        self.caption_type = caption_type

        if transform is None:
            self.transforms = self.build_transform(True, 224, self.mean, self.std)
        else:
            self.transforms = transform
        
        self.df = pd.read_csv(csv_path)
        self.image_files = self.df['img_dir'].tolist()
        #self.captions = self.df['caption'].tolist() 
        self.metadata = self.df.drop(columns=['img_dir', 'caption'])
        
        self.data_len = len(self.df)

    def __getitem__(self, idx):
        
        img_file = self.image_files[idx]
        meta = self.metadata.iloc[idx].to_dict()

        #caption = self.tokenizer(create_fmow_caption(meta, self.caption_type))
        caption = create_fmow_caption(meta, self.caption_type)
        #for key, value in caption.items():
        #    caption[key] = torch.tensor(value)

        # Process image
        img = Image.open(img_file)
        img = self.transforms(img)

        return img, meta, caption

    def __len__(self):
        return self.data_len

def custom_collate_fn(batch):
    images, metas, captions = zip(*batch)
    
    # Stack images (assuming they are already transformed into tensors)
    images = torch.stack(images, 0)
    
    # Combine metas (list of dicts) into a single dict of lists
    #meta_dict = {key: [d[key] for d in metas] for key in metas[0]}
    
    # Captions can be combined into a list
    captions = tokenizer(list(captions), padding=True, truncation=True, return_tensors='pt')
    #for key, value in captions.items():
    #    captions[key] = torch.tensor(value)
    
    #return images, meta_dict, captions
    return images, captions
