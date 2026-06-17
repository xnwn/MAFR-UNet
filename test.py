import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from thop import profile
import torch.fx
from network.cswin_unet import CSWinUnet
from utils.command_line_output import command_line_parameter
from utils.get_dataset import GetDataset
import utils.tester as tester
import os


parser = argparse.ArgumentParser()

parser.add_argument("--save_dir", type=str, required=True, help="Training result save directory")
parser.add_argument("--list_dir", type=str, required=True, help="Sample file name storage directory")
parser.add_argument("--data_dir", type=str, required=True, help="Data storage directory")
parser.add_argument("--log_dir", type=str, required=True, help="Log storage directory")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument("--deterministic", type=int, help="whether use deterministic training")
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

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    command_line_parameter(args)

    dataset = GetDataset(split="test", list_dir=args.list_dir, data_dir=args.data_dir)
    print(f"\nThe length of test set is: {len(dataset)}\n")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

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

    model.load_state_dict(torch.load(f"{args.save_dir}/model.pth"))

    tester.test(args, model, dataset, dataloader)

    inputs = torch.randn(1, args.in_channels, args.img_size, args.img_size).cuda()
    flops, params = profile(model, inputs=(inputs,))
    print(f"Model FLOPs: {flops}")
    print(f"Model Params: {params}\n")
