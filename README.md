# City and Time-Aware Semi-supervised Acoustic Scene Classification

Baseline for APSIPA ASC 2025 GC.

The APSIPA ASC 2025 GC (City and Time-Aware Semi-supervised Acoustic Scene Classification) 
extends the work of the ICME 2024 GC (Semi-supervised Acoustic Scene Classification under Domain Shift), 
which addressed the challenge of generalizing across different cities. 
This year's challenge explicitly incorporates city-level location and timestamp metadata for each audio sample, 
encouraging participants to design models that leverage both geographic and temporal context. 
It maintains the semi-supervised learning setting, reflecting real-world scenarios where large amounts of unlabeled data coexist with limited labeled examples. 
Participants are invited to develop innovative methods that combine audio content with contextual information to enhance classification performance and robustness.

## Challenge Website
[APSIPA ASC 2025 GC](https://www.apsipa2025.org/wp/grand-challenge/)  
[Challenge website](https://ascchallenge.xshengyun.com/)  
[Google groups](https://groups.google.com/g/apsipa2025gc)  
[Development audio recordings](https://zenodo.org/records/10616533)  
[Evaluation audio recordings](https://zenodo.org/records/10820626)  

## Updates

**2025-June-15 The challenge has started and the development dataset is available.**

**2025-July-22 The metadata of the evaluation dataset is available.**

## Official Baseline

![main](pics/APSIPA_2025_ASC_challenge_overview.jpg)

### Step 1: Python Running Environment
```shell
conda create -n ASC python=3.10
conda activate ASC
git clone git@github.com:JishengBai/APSIPA2025GC-ASC.git; cd APSIPA2025GC-ASC
pip install -r requirement.txt
```  

### Step 2: Feature extraction
```shell
# Feature extraction of development dataset:
python3 feature_extraction.py --dataset dev
# Feature extraction of evaluation dataset:
python3 feature_extraction.py --dataset eval
```

### Step3: Train and Evaluate Model

```shell
# Model training, which includes the following three steps:
# (1) Training with limited labels; (2) Pseudo labeling; (3) Model training with pseudo labels.

python train.py

# Model testing, output predicted results of evaluation dataset.
python test.py
```


## Cite
```bibtex
@misc{bai2024description,
      title={Description on IEEE ICME 2024 Grand Challenge: Semi-supervised Acoustic Scene Classification under Domain Shift}, 
      author={Jisheng Bai and Mou Wang and Haohe Liu and Han Yin and Yafei Jia and Siwei Huang and Yutong Du and Dongzhe Zhang and Dongyuan Shi and Woon-Seng Gan and Mark D. Plumbley and Susanto Rahardja and Bin Xiang and Jianfeng Chen},
      year={2024},
      eprint={2402.02694},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
    }
```
```bibtex
@ARTICLE{9951400,
  author={Bai, Jisheng and Chen, Jianfeng and Wang, Mou and Ayub, Muhammad Saad and Yan, Qingli},
  journal={IEEE Transactions on Cognitive and Developmental Systems}, 
  title={A Squeeze-and-Excitation and Transformer-Based Cross-Task Model for Environmental Sound Recognition}, 
  year={2023},
  volume={15},
  number={3},
  pages={1501-1513}
  }
```


## Organization
- Xi'an University of Posts & Telecommunications, China
- Xi'an Lianfeng Acoustic Technologies Co., Ltd., China
- Institute of Acoustics, Chinese Academy of Sciences, China
- University of Surrey, UK
- Northwestern Polytechnical University, China
- Singapore Institute of Technology, Singapore
- Nanyang Technological University, Singapore




