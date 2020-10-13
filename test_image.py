# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import cv2
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from sewar.full_ref import mse
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar.full_ref import vifp

from srgan_pytorch import DatasetFromFolder
from srgan_pytorch import Generator
from srgan_pytorch import cal_niqe
from srgan_pytorch import select_device

parser = argparse.ArgumentParser(description="ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.")
parser.add_argument("--file", type=str, default="./assets/baby.png",
                    help="Test low resolution image name. (default:`./assets/baby.png`)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--model-path", default="./weight/SRGAN_4x.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: ``./weight/SRGAN_4x.pth``).")
parser.add_argument("--device", default="0",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``CUDA:0``).")

args = parser.parse_args()

try:
    os.makedirs("benchmark")
except OSError:
    pass

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=args.batch_size)

dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/train/input",
                            target_dir=f"{args.dataroot}/{args.upscale_factor}x/train/target")

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         pin_memory=True,
                                         num_workers=int(args.workers))

# Construct SRGAN model.
model = Generator(upscale_factor=args.upscale_factor).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))

# Set model eval mode
model.eval()

# Just convert the data to Tensor format
pil2tensor = transforms.ToTensor()

# Load image
img = Image.open(args.file)
hr = pil2tensor(img).unsqueeze(0)
lr = pil2tensor(img).unsqueeze(0)
lr = lr.to(device)

with torch.no_grad():
    sr = model(lr)

vutils.save_image(sr, "sr.png", normalize=False)
vutils.save_image(hr, "hr.png", normalize=False)

# Evaluate performance
src_img = cv2.imread("sr.png")
dst_img = cv2.imread("hr.png")

mse_value = mse(src_img, dst_img)
rmse_value = rmse(src_img, dst_img)
psnr_value = psnr(src_img, dst_img)
ssim_value = ssim(src_img, dst_img)
ms_ssim_value = msssim(src_img, dst_img)
niqe_value = cal_niqe("sr.png")
sam_value = sam(src_img, dst_img)
vif_value = vifp(src_img, dst_img)

print("\n")
print("====================== Performance summary ======================")
print(f"MSE: {mse_value:.2f}\n"
      f"RMSE: {rmse_value:.2f}\n"
      f"PSNR: {psnr_value:.2f}\n"
      f"SSIM: {ssim_value:.4f}\n"
      f"MS-SSIM: {ms_ssim_value:.4f}\n"
      f"NIQE: {niqe_value:.2f}\n"
      f"SAM: {sam_value:.4f}\n"
      f"VIF: {vif_value:.4f}")
print("============================== End ==============================")
print("\n")
