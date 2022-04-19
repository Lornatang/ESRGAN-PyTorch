# ESRGAN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219v2).

## Table of contents

- [ESRGAN-PyTorch](#esrgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train](#train)
        - [Train RRDBNet model](#train-rrdbnet-model)
        - [Train ESRGAN model](#train-esrgan-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](#esrgan-enhanced-super-resolution-generative-adversarial-networks)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the file as follows.

- line 29: `upscale_factor` change to the magnification you need to enlarge.
- line 31: `mode` change Set to valid mode.
- line 104: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 29: `upscale_factor` change to the magnification you need to enlarge.
- line 31: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

### Train RRDBNet model

- line 47: `start_epoch` change number of SRResNet training iterations in the previous round.
- line 48: `resume` change to SRResNet weight address that needs to be loaded.

### Train ESRGAN model

- line 75: `start_epoch` change number of SRGAN training iterations in the previous round.
- line 76: `resume` change to RRDBNet weight address that needs to be loaded.
- line 77: `resume_d` change to Discriminator weight address that needs to be loaded.
- line 78: `resume_g` change to Generator weight address that needs to be loaded.

### Result

Source of original paper results: [https://arxiv.org/pdf/1809.00219v2.pdf](https://arxiv.org/pdf/1809.00219v2.pdf)

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |  RRDBNet (PSNR)  | ESRGAN (PSNR) |
|:-------:|:-----:|:----------------:|:-------------:|
|  Set5   |   4   | 32.73(**32.56**) | -(**29.97**)  |
|  Set14  |   4   | 28.99(**28.48**) | -(**25.61**)  |

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png"/></span>

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks

_Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang_ <br>

**Abstract** <br>
The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image
super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we
thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an
Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network
building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value.
Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency
and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and
natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge. The code is available
at [this https URL](https://github.com/xinntao/ESRGAN).

[[Paper]](https://arxiv.org/pdf/1609.04802) [[Code(PyTorch)]](https://github.com/xinntao/ESRGAN)

```bibtex
@misc{wang2018esrgan,
    title={ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks},
    author={Xintao Wang and Ke Yu and Shixiang Wu and Jinjin Gu and Yihao Liu and Chao Dong and Chen Change Loy and Yu Qiao and Xiaoou Tang},
    year={2018},
    eprint={1809.00219},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
