# MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space
### [Project Page](https://zju3dv.github.io/MotionStreamer/) | [Paper](https://arxiv.org/pdf/2503.15451) | [Data](https://huggingface.co/datasets/lxxiao/272-dim-HumanML3D)
<br/>

> MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space<br>
> [Lixing Xiao](https://li-xingxiao.github.io/homepage/), [Shunlin Lu](https://shunlinlu.github.io/), [Huaijin Pi](https://phj128.github.io/), [Ke Fan](https://vankouf.github.io/), [Liang Pan](https://liangpan99.github.io/), [Yueer Zhou](https://scholar.google.com/citations?user=EWC3UGYAAAAJ/), [Ziyong Feng](https://dblp.org/pid/120/4362.html/), [Xiaowei Zhou](https://www.xzhou.me/), [Sida Peng](https://pengsida.net/)<sup>†</sup>, [Jingbo Wang](https://wangjingbo1219.github.io/)

<img width="1385" alt="image" src="assert/teaser.jpg"/>

## TODO List

- [x] Release the processing script of 272-dim motion representation.
- [x] Release the processed 272-dim Motion Representation of [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset. Only for academic usage.
- [ ] Release complete code for MotionStreamer.

## 🏃 Motion Representation
For more details of how to obtain the 272-dim motion representation, as well as other useful tools (e.g., Visualization and Conversion to BVH format), please refer to our [GitHub repo](https://github.com/Li-xingXiao/272-dim-Motion-Representation).

## 🤗 Processed 272-dim Motion Representation of HumanML3D dataset</b></summary>
To facilitate researchers, we provide the processed 272-dim Motion Representation of [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset on [Hugging Face](https://huggingface.co/datasets/lxxiao/272-dim-HumanML3D).

❗️❗️❗️ The processed data is solely for academic purposes. Make sure you read through the [AMASS License](https://amass.is.tue.mpg.de/license.html).
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download lxxiao/272-dim-HumanML3D --local-dir ./272_humanml3d
cd ./272_humanml3d
unzip texts.zip
unzip motion_data.zip
```
The dataset is organized as:
```
./272_humanml3d
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

## 🌹 Acknowledgement
This repository builds upon the following awesome datasets and projects:
- [AMASS](https://amass.is.tue.mpg.de/index.html)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)

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
