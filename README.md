# ESRGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219).

### Table of contents
1. [About Enhanced Super-Resolution Generative Adversarial Networks](#about-enhanced-super-resolution-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights)
    * [Download dataset](#download-dataset)
4. [Test](#test)
    * [Basic test](#basic-test)
    * [Test benchmark](#test-benchmark)
    * [Test image](#test-image)
4. [Train](#train-eg-div2k)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Enhanced Super-Resolution Generative Adversarial Networks

If you're new to ESRGAN, here's an abstract straight from the paper:

The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge. The code is available at [this https URL](https://github.com/xinntao/ESRGAN) .

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
$ pip install -r requirements.txt
```

#### Download pretrained weights

```bash
$ cd weights/
$ bash download_weights.sh
```

#### Download dataset

```bash
$ cd data/
$ bash download_dataset.sh
```

### Test

Using pre training model to generate pictures.

#### Basic test

```text
usage: test.py [-h] [--dataroot DATAROOT] [-j N] [--image-size IMAGE_SIZE]
               [--scale-factor SCALE_FACTOR] [--cuda] [--weights WEIGHTS]
               [--outf OUTF] [--manualSeed MANUALSEED]

PyTorch Enhance Super Resolution GAN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to dataset. (default:`./data/Set5`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:128)
  --scale-factor SCALE_FACTOR
                        Low to high resolution scaling factor. (default:4).
  --cuda                Enables cuda
  --weights WEIGHTS     Path to weights
                        (default:`./weights/ESRGAN_RRDB_X4.pth`).
  --outf OUTF           folder to output images. (default:`./result`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:0)

# Example
$ python test.py --dataroot ./data/Set5 --cuda --weights ./weights/ESRGAN_RRDB_X4.pth
```

#### Test benchmark

```text
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N]
                         [--image-size IMAGE_SIZE] [--scale-factor {4,8,16}]
                         [--cuda] [--weights WEIGHTS] [--outf OUTF]
                         [--manualSeed MANUALSEED]

PyTorch Enhance Super Resolution GAN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/DIV2K`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:128)
  --scale-factor {4,8,16}
                        Low to high resolution scaling factor. (default:4).
  --cuda                Enables cuda
  --weights WEIGHTS     Path to weights.
                        (default:`./weights/ESRGAN_RRDB_X4.pth`).
  --outf OUTF           folder to output images. (default:`./result`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:0)

# Example
$ python test_benchmark.py --dataroot ./data/DIV2K --cuda --weights ./weights/ESRGAN_RRDB_X4.pth
```

#### Test image

```text
usage: test_image.py [-h] [--file FILE] [--weights WEIGHTS] [--cuda]
                     [--image-size IMAGE_SIZE] [--scale-factor SCALE_FACTOR]

PyTorch Enhance Super Resolution GAN.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution image name.
                        (default:`./assets/baby.png`)
  --weights WEIGHTS     Generator model name.
                        (default:`weights/ESRGAN_RRDB_X4.pth`)
  --cuda                Enables cuda
  --image-size IMAGE_SIZE
                        size of the data crop (squared assumed). (default:128)
  --scale-factor SCALE_FACTOR
                        Super resolution upscale factor

# Example
$ python test_image.py --file ./assets/baby.png --cuda --weights ./weights/ESRGAN_RRDB_X4.pth
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--epochs N]
                [--image-size IMAGE_SIZE] [--scale-factor {4,8,16}] [-b N]
                [--b1 B1] [--b2 B2] [-p N] [--cuda] [--netG NETG]
                [--netD NETD] [--outf OUTF] [--manualSeed MANUALSEED]

PyTorch Enhance Super Resolution GAN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/DIV2K`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --epochs N            Number of total epochs to run. (default:60)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:128)
  --scale-factor {4,8,16}
                        Low to high resolution scaling factor. (default:4).
  -b N, --batch-size N  mini-batch size (default: 16), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --b1 B1               coefficients used for computing running averages of
                        gradient and its square. (default:0.9)
  --b2 B2               coefficients used for computing running averages of
                        gradient and its square. (default:0.999)
  -p N, --print-freq N  Print frequency. (default:5)
  --cuda                Enables cuda
  --netG NETG           Path to netG (to continue training).
  --netD NETD           Path to netD (to continue training).
  --outf OUTF           folder to output images. (default:`./output`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:0)

# Example (e.g DIV2K)
$ python train.py --dataroot ./data/DIV2K --cuda --scale-factor 4
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python train.py --dataroot ./data/DIV2K \
                  --cuda                  \
                  --scale-factor 4        \
                  --netG ./weights/ESRGAN_RRDB_epoch_50.pth \
                  --netD ./weights/ESRGAN_RRDB_epoch_50.pth 
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
_Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang_ <br>

**Abstract** <br>
The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge. The code is available at [this https URL](https://github.com/xinntao/ESRGAN) .

[[Paper]](https://arxiv.org/pdf/1809.00219)

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
