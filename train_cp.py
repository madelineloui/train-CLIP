import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
#from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from vlm_dataloader import FmowDataModule
from transformers import CLIPModel, CLIPTokenizer
from backbones_utils import load_backbone
from util import custom2clip
#from vlm_dataloader import FmowDataset, custom_collate_fn
#import open_clip

# TODO: Correct way to do text projection?
# TODO: Implement for open_clip
# Checkpoint visualizations

def main(hparams):
    
    # Using CLIP pretrained model
    model, tokenizer = load_backbone(hparams.backbone)
    img_encoder = model.vision_model
    txt_encoder = model.text_model
    
    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size
    
    # Init the data module
    dm = FmowDataModule(
        csv_file=hparams.csv_file, 
        tokenizer=tokenizer,
        caption_type=hparams.caption_type, 
        batch_size=hparams.minibatch_size, 
        num_workers=hparams.num_workers
    )

    # Init custom clip model
    model = CustomCLIPWrapper(
        img_encoder,
        txt_encoder,
        hparams.minibatch_size,
        model_name=hparams.model_name,
        avg_word_embs=True,
        warmup_percent=hparams.warmup_percent
    )
    
    # Init checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',        # Metric to monitor
        dirpath=hparams.model_dir,    # Directory to save checkpoints
        filename='model-{epoch:02d}-{val_loss:.2f}',  # Filename template
        save_top_k=3,              # Save the top 3 models with the lowest val_loss
        mode='min',                # 'min' because we want to minimize val_loss
        save_last=True             # Save the model at the last epoch as well
    )
    
    # Set up the CSV logger
    csv_logger = CSVLogger("logs", name=args.exp_name)
    
    # Init trainer
    trainer = Trainer.from_argparse_args(
        hparams, 
        precision=16, 
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        logger = csv_logger
    )
    
    trainer.fit(model, dm)
    
    # Convert ckpts to clip format
    custom2clip(hparams.model_dir, hparams.backbone)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--backbone', type=str, default='clip-14')
    parser.add_argument('--model_name', type=str, default='ViT-L/14') #match model/config
    parser.add_argument('--csv_file', type=str, default='/home/gridsan/manderson/vlm4rs/fmow/val-small.csv')
    parser.add_argument('--caption_type', type=int, default=0)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--warmup_percent', type=float, default=0.10)
    parser.add_argument('--model_dir', type=str, default='/home/gridsan/manderson/train-CLIP/run/test')
    parser.add_argument('--exp_name', type=str, default='test')
    #parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
