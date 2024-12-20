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
#tokenizer = CLIPTokenizer.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")

class FmowDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, csv_file, rgb_path, caption_type=None, caption_path=None, caption_param='', batch_size=32, num_workers=4, val_split=0.2):
        super().__init__()
        global clip_tokenizer
        clip_tokenizer = tokenizer
        self.csv_file = csv_file
        self.rgb_path = rgb_path
        self.caption_type = caption_type
        self.caption_path = caption_path
        self.caption_param = caption_param
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        

    def setup(self, stage=None):
        # Load the full dataset
        dataset = FmowDataset(csv_path=self.csv_file, rgb_path=self.rgb_path, caption_type=self.caption_type, caption_path=self.caption_path, caption_param=self.caption_param)
        
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

    def __init__(self, csv_path, rgb_path, caption_type=None, caption_path=None, caption_param='', transform=None):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__(in_c=3)
        
        self.mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        self.std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        
        self.rgb_path = rgb_path
        self.caption_type = caption_type
        self.caption_path = caption_path
        self.caption_param = caption_param

        if transform is None:
            self.transforms = self.build_transform(True, 224, self.mean, self.std)
        else:
            self.transforms = transform
        
        self.df = pd.read_csv(csv_path)
        #self.image_files = self.df['img_dir'].tolist()
        self.metadata = self.df.drop(columns=['img_id'])
        
        self.data_len = len(self.df)

    def __getitem__(self, idx):
        
        #img_file = self.image_files[idx]
        #meta = self.metadata.iloc[idx].to_dict()

        row = self.df.iloc[idx]
        img_id = row['img_id']
        cat_id = ('_').join(img_id.split('_')[:-2])
        short_id = ('_').join(img_id.split('_')[:-1])
        img_file = f'{self.rgb_path}/{cat_id}/{short_id}/{img_id}_rgb.jpg'
        
        if self.caption_type is not None:
            caption = create_fmow_caption(meta, self.caption_type) 
        else:
            gpt_file = f'{self.caption_path}/{cat_id}/{short_id}_gpt-{self.caption_param}.txt'
            with open(gpt_file, 'r') as file:
                caption = file.read()

        # Process image
        img = Image.open(img_file)
        img = self.transforms(img)

        return img, caption

    def __len__(self):
        return self.data_len

def custom_collate_fn(batch):
    images, captions = zip(*batch)
    
    # Stack images (assuming they are already transformed into tensors)
    images = torch.stack(images, 0)
    
    # Tokenize captions
    captions = clip_tokenizer(list(captions), padding=True, truncation=True, return_tensors='pt')
    
    #return images, captions
    return images, captions
