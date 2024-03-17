<img src="./assets/output.gif">

# 🏃‍♀️NetTrack 

[[`Project`](https://george-zhuang.github.io/nettrack/)] [[`Paper`]()] [[`Demo`](https://www.youtube.com/watch?v=h81R1B8HuOE)] [[`Dataset (Google)`](https://drive.google.com/drive/folders/140mPnOVZY-2apH76at9yYuVGIDWOvsH_?usp=sharing)] [[`Dataset (Baidu)`](https://pan.baidu.com/s/1Ztu8-JJLFHmMkJyWrJQ8lQ?pwd=bft5)] [[`Dataset (Ali)`](https://www.alipan.com/s/NFkpgDDw6R3)]

[Guangze Zheng¹](https://george-zhuang.github.io/), [Shijie Lin¹](https://scholar.google.com/citations?user=sQINQ-YAAAAJ&hl=zh-CN&oi=ao), [Haobo Zuo¹](https://scholar.google.com/citations?user=5RhJGKgAAAAJ&hl=zh-CN&oi=ao), [Changhong Fu²](https://scholar.google.com/citations?user=zmbMZ4kAAAAJ&hl=zh-CN&oi=ao), [Jia Pan¹*](https://scholar.google.com/citations?user=YYT8-7kAAAAJ&hl=zh-CN&oi=ao)

[HKU¹](https://www.hku.hk/), [Tongji University²](https://www.tongji.edu.cn/)

PyTorch implementation for NetTrack.

## 📣 News
- [2024/03/16] 💻 Code has been released.
- [2024/03/01] 📰 NetTrack has been accepted by **CVPR 2024**.

## :hammer_and_wrench: Install 

- **Prerequisite**
  ```bash
  conda create -n nettrack python=3.10 # please use the default version
  pip3 install torch torchvision # --index-url https://download.pytorch.org/whl/cu121
  pip3 install -r requirements.txt
  pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
  pip3 install cython_bbox
  sudo apt update
  sudo apt install ffmpeg
  ```

  Install Grounding DINO and CoTracker:
  ```bash
  pip install git+https://github.com/IDEA-Research/GroundingDINO.git
  pip install git+https://github.com/facebookresearch/co-tracker.git@8d364031971f6b3efec945dd15c468a183e58212
  ```

- **Prepare weights:**
  Download the default pretrained Grouding DINO and CoTracker model:
  ```bash
  cd weights
  cd groundingdino
  wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
  cd ..
  mkdir cotracker && cd cotracker
  wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
  cd ..
  ```

## :bird: BFT dataset
<img src="./assets/dataset_conf.jpg" width="600">

- 📊 Bird flock tracking (**BFT**) dataset:
  - 🎬106 various bird flight videos with 22 species and 14 scenes 
  - 🎯collected for artifical intelligence and ecological research
  - 📈 We provide a Multiple Object Tracking (**MOT**) benchmark for evaluating open-world MOT for highly dynamic object tracking.
  
- 📥 Download **BFT dataset v1.5**
  - **[Recommended]** Download with [Google Drive](https://drive.google.com/drive/folders/140mPnOVZY-2apH76at9yYuVGIDWOvsH_?usp=sharing)
  - Download with [Baidu Pan](https://pan.baidu.com/s/1Ztu8-JJLFHmMkJyWrJQ8lQ?pwd=bft5)
  - Download with [AliPan](https://www.alipan.com/s/NFkpgDDw6R3)
      ```Due to policy limitations of Alipan, please run the .exe file directly to decompress data.```


## 🚀 Get started
- Run default demo video. 
  ```bash
  sh tools/demo/demo_seq.sh
  ```
  The results will be shown in ```./output/track_res```.

- Evaluate
  Please ref to [```./docs/evalutate.md```](./docs/evaluate.md).

## 🎞️ Video
Watch our video on YouTube!

<div class="text-left">
    <iframe width="600" height="330" src="https://www.youtube.com/embed/h81R1B8HuOE?si=6bgoNCkJUEIUjTJY"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen></iframe>
</div>

## 🥰 Acknowledgement
The primary data of BFT dataset is from the BBC nature documentary series [Earthflight](https://www.bbc.co.uk/programmes/b018xsc1). The code is based on [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [CoTracker](https://github.com/facebookresearch/co-tracker), and [ByteTrack](https://github.com/ifzhang/ByteTrack). Dr. [Ming-Shan Wang](https://scholar.google.com/citations?user=bgOWPGwAAAAJ&hl=zh-CN&oi=ao) provided valuable biological suggestions for this work. The authors appreciate the great work and the contributions they made.
## 😮 Cite our work
If you find this dataset useful, please cite our work. Looking forward to your suggestions to make this dataset better!
```
@Inproceedings{nettrack,
title={{NetTrack: Tracking Highly Dynamic Objects with a Net}},
author={Zheng, Guangze and Lin, Shijie and Zuo, Haobo and Fu, Changhong and Pan, Jia},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2024},
pages={1-8}}
```