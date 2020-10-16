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
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N]
                         [--upscale-factor {2,4}] [--model-path PATH]
                         [--device DEVICE]

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weights/ESRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).


# Example
$ python test_benchmark.py --dataroot ./data/DIV2K --upscale-factor 4 --model-path ./weight/ESRGAN_X4.pth --device 0
```

#### Test image

```text
usage: test_image.py [-h] [--lr LR] [--hr HR] [--upscale-factor {2,4}]
                     [--model-path PATH] [--device DEVICE]

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weight/ESRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).

# Example
$ python test_image.py --lr ./lr.png --hr ./hr.png --upscale-factor 4 --model-path ./weight/ESRGAN_X4.pth --device 0
```

#### Test video

```text
usage: test_video.py [-h] --file FILE [--upscale-factor {2,4}]
                     [--model-path PATH] [--device DEVICE] [--view]

ESRGAN algorithm is applied to video files.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default:
                        ``./weight/ESRGAN_4x.pth``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``CUDA:0``).
  --view                Super resolution real time to show.

# Example
$ python test_video.py --file ./lr.mp4 --upscale-factor 4 --model-path ./weight/ESRGAN_X4.pth --device 0
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--start-epoch N]
                [--psnr-iters N] [--iters N] [-b N] [--psnr-lr PSNR_LR]
                [--lr LR] [--upscale-factor {2,4}] [--resume_PSNR] [--resume]
                [--manualSeed MANUALSEED] [--device DEVICE]

ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --start-epoch N       manual epoch number (useful on restarts)
  --psnr-iters N        The number of iterations is needed in the training of
                        PSNR model. (default:1e6)
  --iters N             The training of srgan model requires the number of
                        iterations. (default:4e5)
  -b N, --batch-size N  mini-batch size (default: 16), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --psnr-lr PSNR_LR     Learning rate for PSNR model. (default:2e-4)
  --lr LR               Learning rate. (default:1e-4)
  --upscale-factor {2,4}
                        Low to high resolution scaling factor. (default:4).
  --resume_PSNR         Path to latest checkpoint for PSNR model.
  --resume              Path to latest checkpoint for Generator.
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:10000)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ``).

# Example (e.g DIV2K)
$ python train.py --dataroot ./data/DIV2K --upscale-factor 4
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python train.py --dataroot ./data/DIV2K \
                  --upscale-factor 4        \
                  --resume_PSNR \
                  --resume
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

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
