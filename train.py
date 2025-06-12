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
import sys
import config
import pandas as pd
from dataset import CAS_Dev_Dataset, CAS_unlabel_Dataset
from models.baseline import SE_Trans, ModelTrainer, ModelTester
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchmetrics


def setup_seed(seed):
    """
    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_pretrained_model(model, pretrained_path, remove_fc=False):
    """
    :param model: model
    :param pretrained_path: path of pretrained model
    :param remove_fc: remove fully connected layer
    """
    print(f"== Loading pretrained model from {pretrained_path} ==")
    ckpt = torch.load(pretrained_path)
    
    if remove_fc:
        print("== Removing fc layer parameters due to dimension mismatch ==")
        if "fc.weight" in ckpt["model_state_dict"]:
            ckpt["model_state_dict"].pop("fc.weight")
        if "fc.bias" in ckpt["model_state_dict"]:
            ckpt["model_state_dict"].pop("fc.bias")

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("== Loaded pretrained model (excluding fc layer) ==")
    else:

        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print("== Loaded complete pretrained model ==")

def train(config):
    """
    :param config: configuration module
    """
    os.makedirs(config.output_path, exist_ok=True)
    if os.path.exists(os.path.join(config.output_path, "best_model_1.pth")):
        print("========== First model exists ==========")
        return
    print("========== Training with labeled data ==========")

    dev_meta_csv = pd.read_csv(config.dev_meta_csv_path)
    print("Dev data columns:", dev_meta_csv.columns.tolist())
    print("Unique locations:", dev_meta_csv['location'].unique())
    print("Time format examples:", dev_meta_csv['record_time'].head())
    
    dev_meta_csv = dev_meta_csv.sample(frac=1)
    train_num = int(len(dev_meta_csv) * config.train_val_ratio)
    train_meta_csv = dev_meta_csv[:train_num]
    val_meta_csv = dev_meta_csv[train_num:]
    train_meta_csv.to_csv(
        os.path.join(config.output_path, "train_meta.csv"), index=False
    )
    val_meta_csv.to_csv(os.path.join(config.output_path, "val_meta.csv"), index=False)
    print("== Loading dev dataset ==")
    train_dataset = CAS_Dev_Dataset(config, train_meta_csv, True)
    train_dataloader = train_dataset.get_tensordataset()

    val_dataset = CAS_Dev_Dataset(config, val_meta_csv, False)
    val_dataloader = val_dataset.get_tensordataset()
    print("== Loading pre-trained model ==")
    
    # Load model
    model = SE_Trans(
    frames=config.clip_frames,
    bins=config.n_mels,
    class_num=len(config.class_2_index.keys()),
    nhead=config.nhead,
    dim_feedforward=config.dim_feedforward,
    n_layers=config.n_layers,
    dropout=config.dropout,
    enable_multimodal=config.enable_multimodal,
    num_locations=len(config.location_2_index.keys()),
    location_embedding_dim=config.location_embedding_dim,
    time_feature_dim=config.time_feature_dim,
    time_mapping_dim=config.time_mapping_dim,
    )

    model = model.to(config.device)

    load_pretrained_model(model, config.pretrained_model_path, remove_fc=True)
    print("== Loading model params done ==")

    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step, gamma=config.lr_gamma
    )
    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=len(config.class_2_index.keys()), average="macro", top_k=1
    )
    trainer = ModelTrainer(
        model, criterion, config.device, metric, optimizer, scheduler
    )
    trainer.fit(
        train_data=train_dataloader,
        valid_data=val_dataloader,
        epochs=config.max_epoch,
        exp_path=config.output_path,
        model_file_name="best_model_1.pth",
        early_stopping=config.early_stop_epoch,
    )


def pseudo_labeling(config):
    """
    :param config: configuration module
    """
    train_meta_csv_path = os.path.join(config.output_path, "train_meta.csv")
    val_meta_csv_path = os.path.join(config.output_path, "val_meta.csv")
    if not (os.path.exists(train_meta_csv_path) or os.path.exists(val_meta_csv_path)):
        print("========== Train or val meta csv does not exist ==========")
        sys.exit(0)
    print("========== Pseudo labeling ==========")
    
    print("== Loading model ckpt ==")
    model = SE_Trans(
    frames=config.clip_frames,
    bins=config.n_mels,
    class_num=len(config.class_2_index.keys()),
    nhead=config.nhead,
    dim_feedforward=config.dim_feedforward,
    n_layers=config.n_layers,
    dropout=config.dropout,
    enable_multimodal=config.enable_multimodal,
    num_locations=len(config.location_2_index.keys()),
    location_embedding_dim=config.location_embedding_dim,
    time_feature_dim=config.time_feature_dim,
    time_mapping_dim=config.time_mapping_dim,
    )

    load_pretrained_model(model, os.path.join(config.output_path, "best_model_1.pth"))
    print("== Loading model params done ==")

    ### train pseudo labeling
    print("== Train pseudo labeling ==")
    train_meta_csv = pd.read_csv(train_meta_csv_path, index_col=False)
    train_unlabeled_dataset = CAS_unlabel_Dataset(config, train_meta_csv, "dev")
    train_unlabeled_dataloader = train_unlabeled_dataset.get_tensordataset()

    predicter = ModelTester(model, train_unlabeled_dataloader, config.device)
    train_pse_results = predicter.predict()

    pse_idx = 0
    for index, row in train_meta_csv.iterrows():
        label_str = row["scene_label"]
        if not isinstance(label_str, str):
            train_pse_label_index = train_pse_results[pse_idx][0]
            train_meta_csv.loc[index, "scene_label"] = config.index_2_class[
                train_pse_label_index
            ]
            pse_idx += 1

    ### val pseudo labeling
    print("== Val pseudo labeling ==")
    val_meta_csv = pd.read_csv(val_meta_csv_path, index_col=False)
    val_unlabeled_dataset = CAS_unlabel_Dataset(config, val_meta_csv, "dev")
    val_unlabeled_dataloader = val_unlabeled_dataset.get_tensordataset()

    predicter = ModelTester(model, val_unlabeled_dataloader, config.device)
    val_pse_results = predicter.predict()

    pse_idx = 0
    for index, row in val_meta_csv.iterrows():
        label_str = row["scene_label"]
        if not isinstance(label_str, str):
            val_pse_label_index = val_pse_results[pse_idx][0]
            val_meta_csv.loc[index, "scene_label"] = config.index_2_class[
                val_pse_label_index
            ]
            pse_idx += 1

    train_meta_csv.to_csv(
        os.path.join(config.output_path, "train_meta_pse_labeled.csv"), index=False
    )
    val_meta_csv.to_csv(
        os.path.join(config.output_path, "val_meta_pse_labeled.csv"), index=False
    )


def train_pse(config):
    """
    :param config: configuration module
    """
    os.makedirs(config.output_path, exist_ok=True)
    if os.path.exists(os.path.join(config.output_path, "best_model_2.pth")):
        print("========== Second model exists ==========")
        return
    train_meta_csv_path = os.path.join(config.output_path, "train_meta_pse_labeled.csv")
    val_meta_csv_path = os.path.join(config.output_path, "val_meta_pse_labeled.csv")
    if not (os.path.exists(train_meta_csv_path) or os.path.exists(val_meta_csv_path)):
        print("========== Train or val pse labeled meta csv does not exist ==========")
        sys.exit(0)

    print("========== Pseudo label training ==========")
    train_meta_csv = pd.read_csv(train_meta_csv_path, index_col=False)
    train_dataset = CAS_Dev_Dataset(config, train_meta_csv, True)
    train_dataloader = train_dataset.get_tensordataset()
    val_meta_csv = pd.read_csv(val_meta_csv_path, index_col=False)
    val_dataset = CAS_Dev_Dataset(config, val_meta_csv, False)
    val_dataloader = val_dataset.get_tensordataset()

    # Load model
    print("== Loading model ckpt ==")
    model = SE_Trans(
    frames=config.clip_frames,
    bins=config.n_mels,
    class_num=len(config.class_2_index.keys()),
    nhead=config.nhead,
    dim_feedforward=config.dim_feedforward,
    n_layers=config.n_layers,
    dropout=config.dropout,
    enable_multimodal=config.enable_multimodal,
    num_locations=len(config.location_2_index.keys()),
    location_embedding_dim=config.location_embedding_dim,
    time_feature_dim=config.time_feature_dim,
    time_mapping_dim=config.time_mapping_dim,
    )

    load_pretrained_model(model, os.path.join(config.output_path, "best_model_1.pth"))
    print("== Loading model params done ==")

    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step, gamma=config.lr_gamma
    )
    criterion = nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=len(config.class_2_index.keys()), average="macro", top_k=1
    )
    trainer = ModelTrainer(
        model, criterion, config.device, metric, optimizer, scheduler
    )
    trainer.fit(
        train_data=train_dataloader,
        valid_data=val_dataloader,
        epochs=config.max_epoch,
        exp_path=config.output_path,
        model_file_name="best_model_2.pth",
        early_stopping=config.early_stop_epoch,
    )


if __name__ == "__main__":
    setup_seed(config.random_seed)
    train(config)
    pseudo_labeling(config)
    train_pse(config)
