#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jisheng Bai, Haohe Liu, Han Yin, Mou Wang
@email: baijs@xupt.edu.cn
# Xi'an University of Posts & Telecommunications, China
# Joint Laboratory of Environmental Sound Sensing, School of Marine Science and Technology, Northwestern Polytechnical University, China
# Xi'an Lianfeng Acoustic Technologies Co., Ltd., China
# University of Surrey, UK
# This software is distributed under the terms of the License MIT
"""

import os

# Paths
exp_id = "APSIPA_2025_ASC_baseline"  # exp ID
dev_audio_root_path = "./ICME2024_GC_ASC_dev"
dev_meta_csv_path = "./metadata/APSIPA2025_GC_ASC_dev_metadata.csv"
dev_fea_root_path = "./feature/train"
eval_audio_root_path = "./ICME2024_GC_ASC_eval"
eval_meta_csv_path = "./metadata/APSIPA2025_GC_ASC_eval_metadata.csv"
eval_fea_root_path = "./feature/eval"
pretrained_model_path = "./pretrained/SE-Trans_pretrained_model.pth"
output_path = r"./log/{}".format(exp_id)

os.makedirs("log", exist_ok=True)

###
selected_scene_list = [
    "Bus",
    "Airport",
    "Metro",
    "Restaurant",
    "Shopping mall",
    "Public square",
    "Urban park",
    "Traffic street",
    "Construction site",
    "Bar",
]
class_2_index = {
    "Bus": 0,
    "Airport": 1,
    "Metro": 2,
    "Restaurant": 3,
    "Shopping mall": 4,
    "Public square": 5,
    "Urban park": 6,
    "Traffic street": 7,
    "Construction site": 8,
    "Bar": 9,
}

index_2_class = {
    0: "Bus",
    1: "Airport",
    2: "Metro",
    3: "Restaurant",
    4: "Shopping mall",
    5: "Public square",
    6: "Urban park",
    7: "Traffic street",
    8: "Construction site",
    9: "Bar",
}

# Signal Processing Setting
sample_rate = 44100
clip_frames = 500
n_fft = 2048
win_length = 1764
hop_length = 882
n_mels = 64
fmin = 50
fmax = sample_rate / 2

# Model Setting
device = "cuda:0"
random_seed = 1234
train_val_ratio = 0.8
batch_size = 64
max_epoch = 20
early_stop_epoch = 10
lr = 1e-4
lr_step = 2
lr_gamma = 0.9
nhead = 8
dim_feedforward = 32
n_layers = 1
dropout = 0.1



### multimodal
enable_multimodal = True
location_embedding_dim = 16
time_feature_dim = 8
time_mapping_dim = 16


# city list
location_list = [
    "Xi'an", "Xianyang", "Changchun", "Jinan", "Hefei", "Sanya", "Nanning", "Haikou", "Guilin", "Guangzhou", "Chongqing", 
    "Shenyang", "Beijing", "Baishan", "Taiyuan", "Tianjin", "Nanchang", "Shanghai", "Luoyang", "Liupanshui", "Shangrao", "Dandong"
]

location_2_index = {loc: idx for idx, loc in enumerate(location_list)}
index_2_location = {idx: loc for idx, loc in enumerate(location_list)}
