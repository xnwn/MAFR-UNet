import argparse
import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
import utils.trainer as trainer
import torch.fx

from network.cswin_unet import CSWinUnet
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.command_line_output import command_line_parameter
from utils.get_dataset import GetDataset, RandomGenerator

parser = argparse.ArgumentParser()

parser.add_argument("--save_dir", type=str, required=True, help="Training result save directory")
parser.add_argument("--list_dir", type=str, required=True, help="Sample file name storage directory")
parser.add_argument("--data_dir", type=str, required=True, help="Data storage directory")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument("--deterministic", type=int, help="whether use deterministic training")
parser.add_argument("--batch_size", type=int, help="Batch size per gpu")
parser.add_argument("--pretrain_ckpt_path", type=str, help="Model pre-training weight path")
parser.add_argument("--base_lr", type=float, help="Model base learning rate")
parser.add_argument("--epochs", type=int, help="Epoch number to train")
parser.add_argument("--in_channels", type=int, help="Number of input image channels")
parser.add_argument("--embed_dim", type=int, help="Token embedding dimension")
parser.add_argument("--img_size", type=int, help="The sample size of each image randomly cropped")
parser.add_argument("--drop_path_probability", type=float, help="The DropPath probability in CSWin Transformer Block")
parser.add_argument("--block_num_list", type=str, help="Number of CSWin Transformer Blocks at each stage")
parser.add_argument("--dropout_probability", type=float, help="The Dropout probability in CSWin UNet")
parser.add_argument("--qkv_bias", type=bool, help="Whether to add additive bias when generating qkv")
parser.add_argument("--stripe_width_list", type=str, help="Stripe width values used in each stage")
parser.add_argument("--num_heads_list", type=str, help="Number of attention heads at each stage")
parser.add_argument("--qk_scale", help="Qk result expansion multiples in attention")
parser.add_argument("--mlp_ratio", type=float, help="MLP parameters in CSWin Transformer Block")
parser.add_argument("--num_classes", type=int, help="Number of classes for segmentation results")

args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    command_line_parameter(args)

    dataset = GetDataset(split="train", list_dir=args.list_dir, data_dir=args.data_dir,
                         transforms=transforms.Compose([
                             RandomGenerator(output_size=[args.img_size, args.img_size])
                         ]))
    print(f"\nThe length of train set is: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSWinUnet(in_channels=args.in_channels,
                      embed_dim=args.embed_dim,
                      img_size=args.img_size,
                      block_num_list=eval(args.block_num_list),
                      drop_path_probability=args.drop_path_probability,
                      dropout_probability=args.dropout_probability,
                      qkv_bias=args.qkv_bias,
                      stripe_width_list=eval(args.stripe_width_list),
                      num_heads_list=eval(args.num_heads_list),
                      qk_scale=eval(args.qk_scale),
                      mlp_ratio=args.mlp_ratio,
                      num_classes=args.num_classes)
    model = model.to(device)

    pretrain_ckpt = torch.load(args.pretrain_ckpt_path, map_location=device)
    model.load_state_dict(pretrain_ckpt, strict=False)

    trainer.train(args, model, dataloader, device)
