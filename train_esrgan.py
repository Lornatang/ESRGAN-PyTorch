# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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
import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import esrgan_config
import model
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()
#    print(train_prefetcher[0].shape)
    print("Load all datasets successfully.")

    d_model, g_model, ema_g_model = build_model()
    print(f"Build `{esrgan_config.g_arch_name}` model successfully.")

    pixel_criterion, content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_optimizer(d_model, g_model)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained d model weights...")
    if esrgan_config.pretrained_d_model_weights_path:
        d_model = load_state_dict(d_model, esrgan_config.pretrained_d_model_weights_path)
        print(f"Loaded `{esrgan_config.pretrained_d_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained d model weights not found.")

    print("Check whether to load pretrained g model weights...")
    if esrgan_config.pretrained_g_model_weights_path:
        g_model = load_state_dict(g_model, esrgan_config.pretrained_g_model_weights_path)
        print(f"Loaded `{esrgan_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained g model weights not found.")

    print("Check whether the pretrained d model is restored...")
    if esrgan_config.resume_d_model_weights_path:
        d_model, _, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            d_model,
            esrgan_config.pretrained_d_model_weights_path,
            optimizer=d_optimizer,
            scheduler=d_scheduler,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training d model not found. Start training from scratch.")

    print("Check whether the pretrained g model is restored...")
    if esrgan_config.resume_g_model_weights_path:
        lsrresnet_model, ema_lsrresnet_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            g_model,
            esrgan_config.pretrained_g_model_weights_path,
            ema_model=ema_g_model,
            optimizer=g_optimizer,
            scheduler=g_scheduler,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training g model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", esrgan_config.exp_name)
    results_dir = os.path.join("results", esrgan_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", esrgan_config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(esrgan_config.upscale_factor, esrgan_config.only_test_y_channel)
    ssim_model = SSIM(esrgan_config.upscale_factor, esrgan_config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=esrgan_config.device)
    ssim_model = ssim_model.to(device=esrgan_config.device)

    for epoch in range(start_epoch, esrgan_config.epochs):
        train(d_model,
              g_model,
              ema_g_model,
              train_prefetcher,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler,
              writer)
        psnr, ssim = validate(g_model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              "Test")
        print("\n")

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == esrgan_config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": d_model.state_dict(),
                         "optimizer": d_optimizer.state_dict(),
                         "scheduler": d_scheduler.state_dict()},
                        f"d_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_best.pth.tar",
                        "d_last.pth.tar",
                        is_best,
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": g_model.state_dict(),
                         "ema_state_dict": ema_g_model.state_dict(),
                         "optimizer": g_optimizer.state_dict(),
                         "scheduler": g_scheduler.state_dict()},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset():
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(esrgan_config.train_gt_images_dir,
                                            esrgan_config.train_lr_images_dir)
    test_datasets = TestImageDataset(esrgan_config.test_gt_images_dir, esrgan_config.test_lr_images_dir)
    # for i in train_datasets[0]:
    #     print(train_datasets[0][i].shape,i)
    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=esrgan_config.batch_size,
                                  shuffle=True,
                                  num_workers=esrgan_config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, esrgan_config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, esrgan_config.device)

    return train_prefetcher, test_prefetcher


def build_model():
    d_model = model.__dict__[esrgan_config.d_arch_name]()
    g_model = model.__dict__[esrgan_config.g_arch_name](in_channels=esrgan_config.in_channels,
                                                        out_channels=esrgan_config.out_channels,
                                                        channels=esrgan_config.channels,
                                                        growth_channels=esrgan_config.growth_channels,
                                                        num_blocks=esrgan_config.num_blocks)
    d_model = d_model.to(device=esrgan_config.device)
    g_model = g_model.to(device=esrgan_config.device)

    # Create an Exponential Moving Average Model
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - esrgan_config.model_ema_decay) * averaged_model_parameter + esrgan_config.model_ema_decay * model_parameter
    ema_g_model = AveragedModel(g_model, avg_fn=ema_avg)

    return d_model, g_model, ema_g_model


def define_loss():
    pixel_criterion = nn.L1Loss()
    content_criterion = model.content_loss(esrgan_config.feature_model_extractor_node,
                                           esrgan_config.feature_model_normalize_mean,
                                           esrgan_config.feature_model_normalize_std)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device=esrgan_config.device)
    content_criterion = content_criterion.to(device=esrgan_config.device)
    adversarial_criterion = adversarial_criterion.to(device=esrgan_config.device)

    return pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(d_model, g_model):
    d_optimizer = optim.Adam(d_model.parameters(),
                             esrgan_config.model_lr,
                             esrgan_config.model_betas,
                             esrgan_config.model_eps,
                             esrgan_config.model_weight_decay)
    g_optimizer = optim.Adam(g_model.parameters(),
                             esrgan_config.model_lr,
                             esrgan_config.model_betas,
                             esrgan_config.model_eps,
                             esrgan_config.model_weight_decay)

    return d_optimizer, g_optimizer


def define_scheduler(
        d_optimizer: optim.Adam,
        g_optimizer: optim.Adam
):
    d_scheduler = lr_scheduler.MultiStepLR(d_optimizer,
                                           esrgan_config.lr_scheduler_milestones,
                                           esrgan_config.lr_scheduler_gamma)
    g_scheduler = lr_scheduler.MultiStepLR(g_optimizer,
                                           esrgan_config.lr_scheduler_milestones,
                                           esrgan_config.lr_scheduler_gamma)
    return d_scheduler, g_scheduler


def train(
        d_model: nn.Module,
        g_model: nn.Module,
        ema_g_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        content_criterion: model.content_loss,
        adversarial_criterion: nn.BCEWithLogitsLoss,
        d_optimizer: optim.Adam,
        g_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_gt_probabilities = AverageMeter("D(GT)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              pixel_losses, content_losses, adversarial_losses,
                              d_gt_probabilities, d_sr_probabilities],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    d_model.train()
    g_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)
 #       print(batch_data["lr"].shape, batch_data["gt"].shape)
        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=esrgan_config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=esrgan_config.device, non_blocking=True)

        # Set the real sample label to 1, and the false sample label to 0
        batch_size, _, _, _ = gt.shape
        real_label = torch.full([batch_size, 1], 1.0, dtype=gt.dtype, device=esrgan_config.device)
        fake_label = torch.full([batch_size, 1], 0.0, dtype=gt.dtype, device=esrgan_config.device)

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        with amp.autocast():
            # Use the generator model to generate fake samples
#            print(lr.shape,)
            sr = g_model(lr)
#            print(sr.shape)
            # Output discriminator to discriminate object probability
 #           print(gt.detach().clone().shape)
            gt_output = d_model(gt.detach().clone())
 #           print(gt_output.shape)
            sr_output = d_model(sr)
 #           print(sr_output.shape)
            pixel_loss = esrgan_config.pixel_weight * pixel_criterion(sr, gt)
            content_loss = esrgan_config.content_weight * content_criterion(sr, gt)
            # Computational adversarial network loss
            d_loss_gt = adversarial_criterion(gt_output - torch.mean(sr_output), fake_label) * 0.5
            d_loss_sr = adversarial_criterion(sr_output - torch.mean(gt_output), real_label) * 0.5
            adversarial_loss = esrgan_config.adversarial_weight * (d_loss_gt + d_loss_sr)
            # Calculate the generator total loss value
            g_loss = pixel_loss + content_loss + adversarial_loss
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        scaler.scale(g_loss).backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        scaler.step(g_optimizer)
        scaler.update()

        # Update EMA
        ema_g_model.update_parameters(g_model)
        # Finish training the generator model

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        with amp.autocast():
            gt_output = d_model(gt)
            sr_output = d_model(sr.detach().clone())
            d_loss_gt = adversarial_criterion(gt_output - torch.mean(sr_output), real_label) * 0.5
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        scaler.scale(d_loss_gt).backward(retain_graph=True)

        # Calculate the classification score of the discriminator model for fake samples
        with amp.autocast():
            sr_output = d_model(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output - torch.mean(gt_output), fake_label) * 0.5
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        scaler.scale(d_loss_sr).backward()

        # Calculate the total discriminator loss value
        d_loss = d_loss_gt + d_loss_sr

        # Improve the discriminator model's ability to classify real and fake samples
        scaler.step(d_optimizer)
        scaler.update()
        # Finish training the discriminator model

        # Calculate the score of the discriminator on real samples and fake samples,
        # the score of real samples is close to 1, and the score of fake samples is close to 0
        d_gt_probability = torch.sigmoid_(torch.mean(gt_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))

        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_gt_probabilities.update(d_gt_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % esrgan_config.train_print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", d_gt_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the
        # terminal print data normally
        batch_index += 1


def validate(
        g_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
):
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    g_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=esrgan_config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=esrgan_config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = g_model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % esrgan_config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()
