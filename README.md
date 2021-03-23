# ESRGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219).

### Table of contents
1. [About Enhanced Super-Resolution Generative Adversarial Networks](#about-enhanced-super-resolution-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download dataset](#download-dataset)
4. [Test](#test)
    * [Test benchmark](#test-benchmark)
    * [Test image](#test-image)
    * [Test video](#test-video)
4. [Train](#train-eg-div2k)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Enhanced Super-Resolution Generative Adversarial Networks

If you're new to ESRGAN, here's an abstract straight from the paper:

The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating 
realistic textures during single image super-resolution. However, the hallucinated details are often accompanied 
with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of 
SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive 
an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without 
batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN 
to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the 
perceptual loss by using the features before activation, which could provide stronger supervision for brightness 
consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently 
better visual quality with more realistic and natural textures than SRGAN and won the first place in 
the PIRM2018-SR Challenge. The code is available at [this https URL](https://github.com/xinntao/ESRGAN) .

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. 
It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is 
a discriminant network that discriminates whether an image is real. The input is x, x is a picture, 
and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, 
and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/ESRGAN-PyTorch.git
$ cd ESRGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download dataset

```bash
$ cd data/
$ bash download_dataset.sh
```

### Test

#### Test benchmark

```text
usage: test_benchmark.py [-h] [-a ARCH] [-j N] [-b N] [--sampler-frequency N] [--image-size IMAGE_SIZE] [--upscale-factor {4}] [--model-path PATH] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED]
                         [--gpu GPU] [--multiprocessing-distributed]
                         DIR

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  Model architecture: esrgan16 | esrgan23 | load_state_dict_from_url (default: esrgan16)
  -j N, --workers N     Number of data loading workers. (default: 4)
  -b N, --batch-size N  mini-batch size (default: 16), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --sampler-frequency N
                        If there are many datasets, this method can be used to increase the number of epochs. (default:1)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (default: 128)
  --upscale-factor {4}  Low to high resolution scaling factor. (default: 4)
  --model-path PATH     Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --world-size WORLD_SIZE
                        Number of nodes for distributed training
  --rank RANK           Node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training. (default: tcp://59.110.31.55:12345)
  --dist-backend DIST_BACKEND
                        Distributed backend. (default: nccl)
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training

# Example
$ python3 test_benchmark.py -a esrgan16 --pretrained --gpu 0 [image-folder with train and val folders]
```

#### Test image

```text
usage: test_image.py [-h] --lr LR [--hr HR] [-a ARCH] [--upscale-factor {4}] [--model-path PATH] [--pretrained] [--seed SEED] [--gpu GPU]

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  -a ARCH, --arch ARCH  Model architecture: esrgan16 | esrgan23 | load_state_dict_from_url (default: esrgan16)
  --upscale-factor {4}  Low to high resolution scaling factor. (default: 4)
  --model-path PATH     Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  
# Example
$ python3 test_image.py -a esrgan16 --lr [path-to-lr-image] --hr [Optional, path-to-hr-image] --pretrained --gpu 0
```

#### Test video

```text
usage: test_video.py [-h] --file FILE [-a ARCH] [--upscale-factor {4}] [--model-path PATH] [--pretrained] [--gpu GPU] [--view]

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  -a ARCH, --arch ARCH  Model architecture: esrgan16 | esrgan23 | load_state_dict_from_url (default: srgan)
  --upscale-factor {4}  Low to high resolution scaling factor. (default: 4)
  --model-path PATH     Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --gpu GPU             GPU id to use.
  --view                Do you want to show SR video synchronously.
                        
# Example
$ python3 test_video.py -a esrgan16 --file [path-to-video] --pretrained --gpu 0 --view 
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [-a ARCH] [-j N] [--psnr-epochs N] [--start-psnr-epoch N] [--gan-epochs N] [--start-gan-epoch N] [-b N] [--sampler-frequency N] [--psnr-lr PSNR_LR] [--gan-lr GAN_LR] [--image-size IMAGE_SIZE] [--upscale-factor {4}] [--model-path PATH]
                [--resume_psnr PATH] [--resume_d PATH] [--resume_g PATH] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
                DIR

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  Model architecture: esrgan16 | esrgan23 | load_state_dict_from_url (default: esrgan16)
  -j N, --workers N     Number of data loading workers. (default: 4)
  --psnr-epochs N       Number of total psnr epochs to run. (default: 160)
  --start-psnr-epoch N  Manual psnr epoch number (useful on restarts). (default: 0)
  --gan-epochs N        Number of total gan epochs to run. (default: 64)
  --start-gan-epoch N   Manual gan epoch number (useful on restarts). (default: 0)
  -b N, --batch-size N  mini-batch size (default: 16), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  --sampler-frequency N
                        If there are many datasets, this method can be used to increase the number of epochs. (default:1)
  --psnr-lr PSNR_LR     Learning rate for psnr-oral. (default: 0.0002)
  --gan-lr GAN_LR       Learning rate for gan-oral. (default: 0.0001)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (default: 128)
  --upscale-factor {4}  Low to high resolution scaling factor. (default: 4)
  --model-path PATH     Path to latest checkpoint for model.
  --resume_psnr PATH    Path to latest psnr-oral checkpoint.
  --resume_d PATH       Path to latest -oral checkpoint.
  --resume_g PATH       Path to latest psnr-oral checkpoint.
  --pretrained          Use pre-trained model.
  --world-size WORLD_SIZE
                        Number of nodes for distributed training
  --rank RANK           Node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training. (default: tcp://59.110.31.55:12345)
  --dist-backend DIST_BACKEND
                        Distributed backend. (default: nccl)
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training  
# Example (e.g DIV2K)
$ python train.py -a esrgan16 [image-folder with train and val folders]
# Multi-processing Distributed Data Parallel Training
$ python3 train.py -a esrgan16 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [image-folder with train and val folders]
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a esrgan16 --start-psnr-epoch 10 --resume-psnr weights/PSNR_epoch10.pth [image-folder with train and val folders] 
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
_Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang_ <br>

**Abstract** <br>
The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating 
realistic textures during single image super-resolution. However, the hallucinated details are often accompanied 
with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of 
SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive 
an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without 
batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN 
to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the 
perceptual loss by using the features before activation, which could provide stronger supervision for brightness 
consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently 
better visual quality with more realistic and natural textures than SRGAN and won the first place in 
the PIRM2018-SR Challenge. The code is available at [this https URL](https://github.com/xinntao/ESRGAN) .
[[Paper]](https://arxiv.org/pdf/1609.04802)

```
@misc{wang2018esrgan,
    title={ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks},
    author={Xintao Wang and Ke Yu and Shixiang Wu and Jinjin Gu and Yihao Liu and Chao Dong and Chen Change Loy and Yu Qiao and Xiaoou Tang},
    year={2018},
    eprint={1809.00219},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
