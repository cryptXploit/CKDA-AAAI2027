# Introduction

This is the source code of our AAAI 2026 paper "CKDA: Cross-modality Knowledge Disentanglement and Alignment for Visible-Infrared Lifelong Person Re-identification". Please cite the following paper if you use our code.

Zhenyu Cui, Jiahuan Zhou and Yuxin Peng, "CKDA: Cross-modality Knowledge Disentanglement and Alignment for Visible-Infrared Lifelong Person Re-identification", The 40th AAAI Conference on Artificial Intelligence (AAAI), Singapore, Jan. 20-27, 2026.



# Dependencies

- Python 3.7

- PyTorch 1.13.1



# Data Preparation

## Prepare Datasets
Download the person re-identification datasets [RegDB](http://dm.dongguk.edu/link.html), [SYSU-MM01](https://github.com/wuancong/SYSU-MM01), [LLCM](https://github.com/ZYK100/LLCM), and [HITSZ-VCM](https://github.com/VCM-project233/HITSZ-VCM-data).
Then unzip them and rename them under the directory like
```
Datasets
├── RegDB
│   └──..
├── SYSU-MM01
│   └──..
├── LLCM
│   └──..
├── VCM
    └──..

```
Use ```pre_process_llcm.py```, ```pre_process_sysu.py```, and ```pre_process_vcm.py``` to perform format conversion before training.

# Environment Preparation

- Please follow [DKC](https://github.com/PKU-ICST-MIPL/DKC-CVPR2025) to prepare the execution environment.

# Usage

- Start training by executing the following commands.

- Train&Test: `bash run.sh`


For any questions, feel free to contact us (cuizhenyu@stu.pku.edu.cn).

Welcome to our [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl/home/) for more information about our papers, source codes, and datasets.

