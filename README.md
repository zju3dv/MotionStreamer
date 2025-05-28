
<h2 align="center"<strong>MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space</strong></h2>
  <p align="center">
    <a href='https://li-xingxiao.github.io/homepage/' target='_blank'>Lixing Xiao</a><sup>1</sup>
    ·
    <a href='https://shunlinlu.github.io/' target='_blank'>Shunlin Lu</a> <sup>2</sup>
    ·
    <a href='https://phj128.github.io/' target='_blank'>Huaijin Pi</a><sup>3</sup>
    ·
    <a href='https://vankouf.github.io/' target='_blank'>Ke Fan</a><sup>4</sup>
    ·
    <a href='https://liangpan99.github.io/' target='_blank'>Liang Pan</a><sup>3</sup>
    ·
    <a href='https://scholar.google.com/citations?user=EWC3UGYAAAAJ/' target='_blank'>Yueer Zhou</a><sup>1</sup>
    ·
    <a href='https://dblp.org/pid/120/4362.html/' target='_blank'>Ziyong Feng</a><sup>5</sup>
    ·
    <br>
    <a href='https://www.xzhou.me/' target='_blank'>Xiaowei Zhou</a><sup>1</sup>
    ·
    <a href='https://pengsida.net/' target='_blank'>Sida Peng</a><sup>1†</sup>
    ·
     <a href='https://wangjingbo1219.github.io/' target='_blank'>Jingbo Wang</a><sup>6</sup>
    <br>
    <br>
    <sup>1</sup>Zhejiang University  <sup>2</sup>The Chinese University of Hong Kong, Shenzhen  <sup>3</sup>The University of Hong Kong  <br><sup>4</sup>Shanghai Jiao Tong University  <sup>5</sup>DeepGlint  <sup>6</sup>Shanghai AI Lab
    <br>
    <strong>Arxiv 2025</strong>
    
  </p>
</p>
<p align="center">
  <a href='https://arxiv.org/abs/2503.15451'>
    <img src='https://img.shields.io/badge/Arxiv-2503.15451-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://arxiv.org/pdf/2503.15451'>
    <img src='https://img.shields.io/badge/Paper-PDF-blue?style=flat&logo=arXiv&logoColor=blue'></a>
  <a href='https://zju3dv.github.io/MotionStreamer/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://huggingface.co/datasets/lxxiao/272-dim-HumanML3D'>
    <img src='https://img.shields.io/badge/Data-Download-yellow?style=flat&logo=huggingface&logoColor=yellow'></a>
</p>

<img width="1385" alt="image" src="assert/teaser.jpg"/>

## TODO List

- [x] Release the processing script of 272-dim motion representation.
- [x] Release the processed 272-dim Motion Representation of [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset. Only for academic usage.
- [x] Release the training code and checkpoint of our [TMR](https://github.com/Mathux/TMR)-based motion evaluator trained on the processed 272-dim [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset.
- [x] Release the training and evaluation code of Causal TAE.
- [ ] Release complete code for MotionStreamer.

## 🏃 Motion Representation
For more details of how to obtain the 272-dim motion representation, as well as other useful tools (e.g., Visualization and Conversion to BVH format), please refer to our [GitHub repo](https://github.com/Li-xingXiao/272-dim-Motion-Representation).

## Installation

### 🐍 Python Virtual Environment
```sh
conda env create -f environment.yaml
conda activate mgpt
```

### 🤗 Hugging Face Mirror
Since all of our models and data are available on Hugging Face, if Hugging Face is not directly accessible, you can use the HF-mirror tools following:
```sh
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```

## 📥 Data Preparation
To facilitate researchers, we provide the processed 272-dim Motion Representation of [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset on [Hugging Face](https://huggingface.co/datasets/lxxiao/272-dim-HumanML3D).

❗️❗️❗️ The processed data is solely for academic purposes. Make sure you read through the [AMASS License](https://amass.is.tue.mpg.de/license.html).
Download the processed 272-dim [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset following:
```bash
huggingface-cli download --repo-type dataset --resume-download lxxiao/272-dim-HumanML3D --local-dir ./humanml3d_272
cd ./humanml3d_272
unzip texts.zip
unzip motion_data.zip
```
The dataset is organized as:
```
./humanml3d_272
  ├── mean_std
      ├── Mean.npy
      ├── Std.npy
  ├── split
      ├── train.txt
      ├── val.txt
      ├── test.txt
  ├── texts
      ├── 000000.txt
      ...
  ├── motion_data
      ├── 000000.npy
      ...
```

## 🚀 Training
1. Train our [TMR](https://github.com/Mathux/TMR)-based motion evaluator on the processed 272-dim [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset following:
    ```bash
    bash TRAIN_evaluator_272.sh
    ```
    >After training for 100 epochs, the checkpoint will be stored at: 
    ``Evaluator_272/experiments/temos/EXP1/checkpoints/``.

    We provide the evaluator checkpoint on [Hugging Face](https://huggingface.co/lxxiao/MotionStreamer/tree/main/Evaluator_272), download it following:
    ```bash
    python humanml3d_272/prepare/download_evaluator_ckpt.py
    ```
    >The downloaded checkpoint will be stored at: ``Evaluator_272/``.
2. Train the Causal TAE following:
    ```bash
      bash TRAIN_causal_TAE.sh ${NUM_GPUS}
    ```
    > e.g., if you have 8 GPUs, run: bash TRAIN_causal_TAE.sh 8

    > The checkpoint will be stored at:
    ``output/causal_TAE/``

    > Tensorboard visualization:
    ```bash
      tensorboard --logdir='output/causal_TAE'
    ```

## 📍 Evaluation

1. Evaluate the metrics of the processed 272-dim [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset following:
    ```bash
    bash EVAL_GT.sh
    ```
    ( FID, R@1, R@2, R@3, Diversity and MM-Dist (Matching Score) are reported. )

2. Evaluate the metrics of Causal TAE following:
    ```bash
      bash EVAL_causal_TAE.sh
    ```
    ( FID and MPJPE (mm) are reported. )
  

## 🌹 Acknowledgement
This repository builds upon the following awesome datasets and projects:
- [AMASS](https://amass.is.tue.mpg.de/index.html)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- [T2M-GPT](https://github.com/Mael-zys/T2M-GPT)
- [TMR](https://github.com/Mathux/TMR)
- [OpenTMA](https://github.com/LinghaoChan/OpenTMA)
- [Sigma-VAE](https://github.com/orybkin/sigma-vae-pytorch)

## 🤝🏼 Citation
If our project is helpful for your research, please consider citing :
``` 
@article{xiao2025motionstreamer,
      title={MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space},
      author={Xiao, Lixing and Lu, Shunlin and Pi, Huaijin and Fan, Ke and Pan, Liang and Zhou, Yueer and Feng, Ziyong and Zhou, Xiaowei and Peng, Sida and Wang, Jingbo},
      journal={arXiv preprint arXiv:2503.15451},
      year={2025}
    }
```
