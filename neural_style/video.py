import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import zipfile
import utils
from transformer_net import TransformerNet
from vgg import Vgg16



"""
python neural_style/neural_style.py eval 
--content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
"""

"""
python /code/fast_neural_style/neural_style/neural_style.py train --dataset '/data/SHRIMP525/coco_2014/' --style-image '/code/fast_neural_style/images/style-images/yinghua.jpg' --save-model-dir '/output/yinghua.pth' --epochs 2 --cuda 1


python /code/fast_neural_style/neural_style/neural_style.py train 
--dataset '/data/SHRIMP525/coco_2014/train2014' --style-image '/code/fast_neural_style/images/style-images/yinghua.jpg' 
--save-model-dir '/output/yinghua.pth' --epochs 2 --cuda 1




"""

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)



"""
1~344
"""
def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    for i in range(344):
        j=i+1
        image_pth = args.content_image + '\\' + str(j) + '.jpeg'
        content_image = utils.load_image(image_pth, scale=args.content_scale)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        if args.model.endswith(".onnx"):
            return
        else:
            with torch.no_grad():
                style_model = TransformerNet()
                state_dict = torch.load(args.model)
                # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
                for k in list(state_dict.keys()):
                    if re.search(r'in\d+\.running_(mean|var)$', k):
                        del state_dict[k]
                style_model.load_state_dict(state_dict)
                style_model.to(device)
                if args.export_onnx:
                    assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                    output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
                else:
                    output = style_model(content_image).cpu()
        save_img = args.output_image + '\\' + str(j) + '.jpg'
        utils.save_image(save_img, output[0])




def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
    else:
        stylize(args)


"""
model: candy  mosaic  rain_princess  udnie
"""


if __name__ == "__main__":
    main()
