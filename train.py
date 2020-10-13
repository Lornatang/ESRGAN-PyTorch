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
import csv
import logging
import os
import random

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from tqdm import tqdm

from esrgan_pytorch import DatasetFromFolder
from esrgan_pytorch import Discriminator
from esrgan_pytorch import FeatureExtractorVGG54
from esrgan_pytorch import Generator
from esrgan_pytorch import init_torch_seeds
from esrgan_pytorch import load_checkpoint
from esrgan_pytorch import select_device

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("--psnr_iters", default=1e6, type=int, metavar="N",
                    help="The number of iterations is needed in the training of PSNR model. (default:1e6)")
parser.add_argument("--iters", default=4e5, type=int, metavar="N",
                    help="The training of srgan model requires the number of iterations. (default:4e5)")
parser.add_argument("-b", "--batch-size", default=16, type=int, metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--psnr-lr", type=float, default=2e-4,
                    help="Learning rate for PSNR model. (default:2e-4)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate. (default:1e-4)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--resume_PSNR", action="store_true",
                    help="Path to latest checkpoint for PSNR model.")
parser.add_argument("--resume", action="store_true",
                    help="Path to latest checkpoint for Generator.")
parser.add_argument("--manualSeed", type=int, default=10000,
                    help="Seed for initializing training. (default:10000)")
parser.add_argument("--device", default="",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``).")

args = parser.parse_args()
print(args)

try:
    os.makedirs("output/lr")
    os.makedirs("output/hr")
    os.makedirs("output/sr")
    os.makedirs("weight")
except OSError:
    logger.error("[!] Failed to create folder because the folder already exists!")
    pass

# Set random initialization seed, easy to reproduce.
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
init_torch_seeds(args.manualSeed)

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=args.batch_size)

dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/train/input",
                            target_dir=f"{args.dataroot}/{args.upscale_factor}x/train/target")

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         pin_memory=True,
                                         num_workers=int(args.workers))

# Construct network architecture model of generator and discriminator.
netG = Generator(upscale_factor=args.upscale_factor).to(device)
netD = Discriminator().to(device)

# Define PSNR model optimizers
psnr_epochs = int(args.psnr_iters // len(dataloader))
epoch_indices = int(psnr_epochs // 4)
optimizer = optim.Adam(netG.parameters(), lr=args.psnr_lr, betas=(0.9, 0.99))
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                           T_0=epoch_indices,
                                                           T_mult=1,
                                                           eta_min=1e-7)

# Loading PSNR pre training model
if args.resume_PSNR:
    args.start_epoch = load_checkpoint(netG, optimizer, f"./weight/RRDBNet_PSNR_{args.upscale_factor}x_checkpoint.pth")

# We use vgg54 as our feature extraction method by default.
feature_extractor = FeatureExtractorVGG54().to(device)
# TODO: Loss = content_loss + 2e-6 * content_loss + 1e-3 * adversarial_loss
content_criterion = nn.L1Loss().to(device)
adversarial_criterion = nn.BCELoss().to(device)

# Set the all model to training mode
netG.train()
netD.train()
feature_extractor.train()

# Pre-train generator using raw MSE loss
logger.info("[*] Start training PSNR model based on MSE loss.")
logger.info(f"[*] Generator pre-training for {psnr_epochs} epochs.")
logger.info(f"[*] Searching PSNR pretrained model weights.")

# Writer train PSNR model log.
if args.start_epoch == 0:
    with open(f"RRDBNet_PSNR_{args.upscale_factor}x_Loss.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "MSE Loss"])

# Save the generator model based on MSE pre training to speed up the training time
if os.path.exists(f"./weight/RRDBNet_PSNR_{args.upscale_factor}x_checkpoint.pth"):
    logger.info("[*] Found PSNR pretrained model weights. Skip pre-train.")
    netG.load_state_dict(torch.load(f"./weight/RRDBNet_PSNR_{args.upscale_factor}x_checkpoint.pth",
                                    map_location=device))
else:
    logger.warning("[!] Not found pretrained weights. Start training PSNR model.")
    for epoch in range(args.start_epoch, psnr_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        avg_loss = 0.
        for i, (input, target) in progress_bar:
            # Set generator gradients to zero
            netG.zero_grad()
            # Generate data
            lr = input.to(device)
            hr = target.to(device)

            # Generating fake high resolution images from real low resolution images.
            sr = netG(lr)
            # The MSE of the generated fake high-resolution image and real high-resolution image is calculated.
            loss = content_criterion(sr, hr)
            # Calculate gradients for generator
            loss.backward()
            # Update generator weights
            optimizer.step()

            avg_loss += loss.item()

            progress_bar.set_description(f"[{epoch + 1}/{psnr_epochs}][{i + 1}/{len(dataloader)}] "
                                         f"MSE: {loss.item():.4f}")

            # The image is saved every 5000 iterations.
            if (len(dataloader) * epoch + i + 1) % 5000 == 0:
                vutils.save_image(lr, f"./output/lr/RRDBNet_PSNR_{len(dataloader) * epoch + i + 1}.bmp", normalize=True)
                vutils.save_image(hr, f"./output/hr/RRDBNet_PSNR_{len(dataloader) * epoch + i + 1}.bmp", normalize=True)
                vutils.save_image(sr, f"./output/sr/RRDBNet_PSNR_{len(dataloader) * epoch + i + 1}.bmp", normalize=True)

        # The model is saved every 1 epoch.
        torch.save({"epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": netG.state_dict()
                    }, f"./weight/RRDBNet_PSNR_{args.upscale_factor}x_checkpoint.pth")

        # Writer training log
        with open(f"RRDBNet_PSNR_{args.upscale_factor}x_Loss.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss / len(dataloader)])

    torch.save(netG.state_dict(), f"./weight/RRDBNet_PSNR_{args.upscale_factor}x_checkpoint.pth")
    logger.info(f"[*] Training PSNR model done! Saving PSNR model weight to "
                f"`./weight/RRDBNet_PSNR_{args.upscale_factor}x_checkpoint.pth`")

# Alternating training SRGAN network.
epochs = int(args.iters // len(dataloader))
epoch_indices = [int(50000 // len(dataloader)), int(100000 // len(dataloader)),
                 int(200000 // len(dataloader)), int(300000 // len(dataloader))]
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.99))
schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=epoch_indices, gamma=0.5)
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=epoch_indices, gamma=0.5)

# Loading SRGAN checkpoint
if args.resume:
    args.start_epoch = load_checkpoint(netD, optimizerD, f"./weight/netD_{args.upscale_factor}x_checkpoint.pth")
    args.start_epoch = load_checkpoint(netG, optimizerG, f"./weight/netG_{args.upscale_factor}x_checkpoint.pth")

# Train SRGAN model.
logger.info(f"[*] Staring training ESRGAN model!")
logger.info(f"[*] Training for {epochs} epochs.")
# Writer train PSNR model log.
if args.start_epoch == 0:
    with open(f"ESRGAN_{args.upscale_factor}x_Loss.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "D Loss", "G Loss"])

for epoch in range(args.start_epoch, epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    g_avg_loss = 0.
    d_avg_loss = 0.
    for i, (input, target) in progress_bar:
        lr = input.to(device)
        hr = target.to(device)
        batch_size = lr.size(0)
        real_label = torch.full((batch_size,), 1, dtype=lr.dtype, device=device)
        fake_label = torch.full((batch_size,), 0, dtype=lr.dtype, device=device)

        ##############################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ##############################################
        # Set discriminator gradients to zero.
        netD.zero_grad()

        # Train with real high resolution image.
        hr_output = netD(hr)
        errD_hr = adversarial_criterion(hr_output, real_label)
        errD_hr.backward()
        D_x = hr_output.mean().item()

        # First train with fake high resolution image.
        sr = netG(lr)
        sr_output = netD(sr.detach())
        errD_sr = adversarial_criterion(sr_output, fake_label)
        errD_sr.backward()
        # Score of generator's first high resolution image.
        D_G_z1 = sr_output.mean().item()
        errD = errD_hr + errD_sr
        optimizerD.step()

        d_avg_loss += errD.item()

        ##############################################
        # (2) Update G network: maximize log(D(G(z)))
        ##############################################
        # Set generator gradients to zero
        netG.zero_grad()

        # Pixel level loss between two images.
        mse_loss = content_criterion(sr, lr)

        # According to the feature map, the root mean square error is regarded as the content loss.
        vgg_loss = content_criterion(feature_extractor(sr), feature_extractor(hr))

        # Second train with fake high resolution image.
        sr_output = netD(sr)
        adversarial_loss = adversarial_criterion(sr_output, real_label)

        # The original content loss in this paper consists of MSE loss and VGG loss.
        content_loss = mse_loss + 6e-3 * vgg_loss

        # # The original perceptual loss in this paper consists of VGG loss, MSE loss and adversarial loss.
        errG = content_loss + 1e-3 * adversarial_loss
        errG.backward()
        D_G_z2 = sr_output.mean().item()
        optimizerG.step()

        # Dynamic adjustment of learning rate
        schedulerG.step()
        schedulerD.step()

        g_avg_loss += errG.item()

        progress_bar.set_description(f"[{epoch + 1}/{args.epochs}][{i + 1}/{len(dataloader)}] "
                                     f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                                     f"D(x): {D_x:.4f} D(G(lr)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        # The image is saved every 5000 iterations.
        if (len(dataloader) * epoch + i + 1) % 5000 == 0:
            vutils.save_image(lr, f"./output/lr/ESRGAN_{len(dataloader) * epoch + i + 1}.bmp", normalize=True)
            vutils.save_image(hr, f"./output/hr/ESRGAN_{len(dataloader) * epoch + i + 1}.bmp", normalize=True)
            vutils.save_image(sr, f"./output/sr/ESRGAN_{len(dataloader) * epoch + i + 1}.bmp", normalize=True)

        # The model is saved every 1 epoch.
        torch.save({"epoch": epoch + 1,
                    "optimizer": optimizerD.state_dict(),
                    "state_dict": netD.state_dict()
                    }, f"./weight/netD_{args.upscale_factor}x_checkpoint.pth")
        torch.save({"epoch": epoch + 1,
                    "optimizer": optimizerG.state_dict(),
                    "state_dict": netG.state_dict()
                    }, f"./weight/netG_{args.upscale_factor}x_checkpoint.pth")

        # Writer training log
        with open(f"ESRGAN_{args.upscale_factor}x_Loss.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, d_avg_loss / len(dataloader), g_avg_loss / len(dataloader)])

    torch.save(netG.state_dict(), f"./weight/ESRGAN_{args.upscale_factor}x.pth")
    logger.info(f"[*] Training SRGAN model done! Saving SRGAN model weight "
                f"to `./weight/ESRGAN_{args.upscale_factor}x.pth`.")
