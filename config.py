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
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "train_rrdbnet"
# Experiment name, easy to save weights and log files
exp_name = "RRDBNet_baseline"

if mode == "train_rrdbnet":
    # Dataset address
    train_image_dir = "data/DFO2K/ESRGAN/train"
    valid_image_dir = "data/DFO2K/ESRGAN/valid"
    test_image_dir = "data/Set5/GTmod12"

    image_size = 192
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total num epochs
    epochs = 20

    # Adam optimizer parameter for RRDBNet(p)
    model_lr = 2e-4
    model_betas = (0.9, 0.999)

    # StepLR scheduler
    lr_scheduler_step_size = epochs // 5
    lr_scheduler_gamma = 0.5

    print_frequency = 100

if mode == "train_esrgan":
    # Dataset address
    train_image_dir = "data/DFO2K/ESRGAN/train"
    valid_image_dir = "data/DFO2K/ESRGAN/valid"
    test_image_dir = "data/Set5/GTmod12"

    image_size = 128
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = "results/RRDBNet_baseline/g_best.pth.tar"
    resume_d = ""
    resume_g = ""

    # Total num epochs
    epochs = 8

    # Loss function weight
    pixel_weight = 1.0
    content_weight = 1.0
    adversarial_weight = 0.001

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # MultiStepLR scheduler parameter
    lr_scheduler_milestones = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]
    lr_scheduler_gamma = 0.5

    print_frequency = 100

if mode == "valid":
    # Test data address
    lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    model_path = f"results/{exp_name}/g_best.pth.tar"
