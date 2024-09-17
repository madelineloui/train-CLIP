import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
#from data.text_image_dm import TextImageDataModule
from models import CustomCLIPWrapper
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from vlm_dataloader import FmowDataModule
#from vlm_dataloader import FmowDataset, custom_collate_fn
import open_clip
from transformers import CLIPModel, CLIPTokenizer

# TODO: Input the model name (vit-L) as a param
# TODO: Add my own dataloader

def main(hparams):
    
    '''
    # Image and text pretrained encoders
    model, _, _ = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pretrained)
    model.output_tokens = True
    img_encoder = model.visual
    txt_encoder = model.transformer
    tokenizer = open_clip.get_tokenizer(args.backbone)
    '''
    model = CLIPModel.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
    img_encoder = model.vision_model
    txt_encoder = model.text_model
    print(img_encoder)
    print(txt_encoder)
    tokenizer = CLIPTokenizer.from_pretrained("/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14")
    
    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size
    
    '''
    # fmow dataset
    fmow_dataset = FmowDataset(hparams.csv_file, hparams.caption_type)
    dm = DataLoader(
        fmow_dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=hparams.num_workers, pin_memory=True)
    '''
    
    # Initialize the data module
    dm = FmowDataModule(
        csv_file=hparams.csv_file, 
        tokenizer=tokenizer,
        caption_type=hparams.caption_type, 
        batch_size=hparams.minibatch_size, 
        num_workers=hparams.num_workers
    )

    model = CustomCLIPWrapper(img_encoder, txt_encoder, hparams.minibatch_size, avg_word_embs=True)
    
    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=50)
    trainer.fit(model, dm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--backbone', type=str, default='ViT-L-14')
    parser.add_argument('--pretrained', type=str, default='openai')
    parser.add_argument('--csv_file', type=str, default='/home/gridsan/manderson/vlm4rs/fmow/train.csv')
    parser.add_argument('--caption_type', type=int, default=0)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=16)
    #parser.add_argument('--max_epochs', type=int, default=50)
    #parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
