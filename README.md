# ESRGAN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219v2).

## Table of contents

- [ESRGAN-PyTorch](#esrgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train RRDBNet model](#train-rrdbnet-model)
        - [Resume train RRDBNet model](#resume-train-rrdbnet-model)
        - [Train ESRGAN model](#train-esrgan-model)
        - [Resume train ESRGAN model](#resume-train-esrgan-model)
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

## How Test and Train

Both training and testing only need to modify the `config.py` file. 

### Test

- line 31: `upscale_factor` change to `4`.
- line 33: `mode` change to `test`.
- line 111: `model_path` change to `results/pretrained_models/RRDBNet_x4-DFO2K-2e2a91f4.pth.tar`.

### Train RRDBNet model

- line 31: `upscale_factor` change to `4`.
- line 33: `mode` change to `train_rrdbnet`.
- line 35: `exp_name` change to `RRDBNet_baseline`.

### Resume train RRDBNet model

- line 31: `upscale_factor` change to `4`.
- line 33: `mode` change to `train_rrdbnet`.
- line 35: `exp_name` change to `RRDBNet_baseline`.
- line 49: `resume` change to `samples/RRDBNet_baseline/g_epoch_xxx.pth.tar`.

### Train ESRGAN model

- line 31: `upscale_factor` change to `4`.
- line 33: `mode` change to `train_esrgan`.
- line 35: `exp_name` change to `ESRGAN_baseline`.
- line 77: `resume` change to `results/RRDBNet_baseline/g_last.pth.tar`.

### Resume train ESRGAN model

- line 31: `upscale_factor` change to `4`.
- line 33: `mode` change to `train_esrgan`.
- line 35: `exp_name` change to `ESRGAN_baseline`.
- line 77: `resume` change to `results/RRDBNet_baseline/g_last.pth.tar`.
- line 78: `resume_d` change to `samples/ESRGAN_baseline/g_epoch_xxx.pth.tar`.
- line 79: `resume_g` change to `samples/ESRGAN_baseline/g_epoch_xxx.pth.tar`.

### Result

Source of original paper results: [https://arxiv.org/pdf/1809.00219v2.pdf](https://arxiv.org/pdf/1809.00219v2.pdf)

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Method | Scale |          Set5 (PSNR/SSIM)           |          Set14 (PSNR/SSIM)          |         BSD100 (PSNR/SSIM)          |        Urban100 (PSNR/SSIM)         |        Manga109 (PSNR/SSIM)         |
|:------:|:-----:|:-----------------------------------:|:-----------------------------------:|:-----------------------------------:|:-----------------------------------:|:-----------------------------------:|
|  RRDB  |   4   | 32.73(**32.71**)/0.9011(**0.9018**) | 28.99(**28.96**)/0.7917(**0.7917**) | 27.85(**27.85**)/0.7455(**0.7473**) | 28.03(**28.03**)/0.8153(**0.8156**) | 31.66(**31.60**)/0.9196(**0.9195**) |
| ESRGAN |   4   |     -(**30.44**)/-(**0.8525**)      |     -(**26.28**)/-(**0.6994**)      |     -(**25.33**)/-(**0.6534**)      |     -(**24.36**)/-(**0.7341**)      |     -(**29.42**)/-(**0.8597**)      |

```bash
# Download `ESRGAN_x4-DFO2K-25393df7.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python ./inference.py --inputs_path ./figure/baboon_lr.png --output_path ./figure/baboon_sr.png --weights_path ./results/pretrained_models/ESRGAN_x4-DFO2K-25393df7.pth.tar
```

Input: 

<span align="center"><img width="492" height="480" src="figure/baboon_lr.png"/></span>

Output: 

<span align="center"><img width="492" height="480" src="figure/baboon_sr.png"/></span>

```text
Build ESRGAN model successfully.
Load ESRGAN model weights `./results/pretrained_models/ESRGAN_x4-DFO2K-25393df7.pth.tar` successfully.
SR image save to `./figure/baboon_sr.png`
```

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

[[Paper]](https://arxiv.org/pdf/1809.00219v2.pdf) [[Author's implements(PyTorch)]](https://github.com/xinntao/ESRGAN)

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
