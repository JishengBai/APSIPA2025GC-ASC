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
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset

from datetime import datetime

def extract_time_features(time_str):
    '''
    time metadata coding
    '''
    try:
        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M")
        
        hour = dt.hour
        month = dt.month
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        minute = dt.minute
        
        # sin-cos coding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        weekday_sin = np.sin(2 * np.pi * weekday / 7)
        weekday_cos = np.cos(2 * np.pi * weekday / 7)
        minute_sin = np.sin(2 * np.pi * minute / 60)
        minute_cos = np.cos(2 * np.pi * minute / 60)
        
        return np.array([hour_sin, hour_cos, month_sin, month_cos, 
                        weekday_sin, weekday_cos, minute_sin, minute_cos], dtype=np.float32)
    except:

        return np.zeros(8, dtype=np.float32)

class CAS_Dev_Dataset(object):
    def __init__(self, data_config, data_csv, is_train: bool):
        """
        :param data_config: configuration module
        :param data_csv: metadata dataframe
        :param is_train: True/False for training/validation data
        """
        self.stats_csv = data_csv
        self.root_path = data_config.dev_fea_root_path
        self.selected_scene_list = data_config.selected_scene_list
        self.tar_sr = data_config.sample_rate
        self.batch_size = data_config.batch_size
        self.clip_frames = data_config.clip_frames
        self.is_train = is_train
        self.class_2_index = data_config.class_2_index
        self.enable_multimodal = getattr(data_config, 'enable_multimodal', False)
        self.location_2_index = getattr(data_config, 'location_2_index', {})

        self.file_list = []
        self.label_list = []
        self.location_list = []
        self.time_feature_list = []
        self.get_file_list()

    def get_file_list(self):
        selected_data = self.stats_csv[
            self.stats_csv["scene_label"].isin(self.selected_scene_list)
        ]

        for index, row in selected_data.iterrows():
            label_str = row["scene_label"]

            if isinstance(label_str, str):
                filename = row["filename"]
                file_path = os.path.join(self.root_path, filename + ".npy")
                self.file_list.append(file_path)
                self.label_list.append(label_str)
                
                if self.enable_multimodal:
                    # extract city info
                    location = row.get("location", "unknown")
                    location_idx = self.location_2_index.get(location, 0)  # 默认为0
                    self.location_list.append(location_idx)
                    
                    # extract time info
                    record_time = row.get("record_time", "")
                    time_features = extract_time_features(record_time)
                    self.time_feature_list.append(time_features)

    def get_numpy_dataset(self):
        data = []
        label = []
        locations = []
        time_features = []
        
        for file_path in tqdm(self.file_list):
            file = np.load(file_path, allow_pickle=True)
            file = file[: self.clip_frames, :]
            data.append(file)

        for label_str in tqdm(self.label_list):
            lb = self.class_2_index[label_str]
            label.append(lb)

        data = np.asarray(data)
        label = np.asarray(label, dtype=np.int32)
        
        if self.enable_multimodal:
            locations = np.asarray(self.location_list, dtype=np.int32)
            time_features = np.asarray(self.time_feature_list, dtype=np.float32)

        return data, label, locations, time_features

    def get_tensordataset(self):
        data, label, locations, time_features = self.get_numpy_dataset()
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(label).long()
        
        if self.enable_multimodal and len(locations) > 0:
            location_tensor = torch.from_numpy(locations).long()
            time_feature_tensor = torch.from_numpy(time_features).float()
            dataset = TensorDataset(data_tensor, label_tensor, location_tensor, time_feature_tensor)
        else:
            location_tensor = torch.zeros(len(data_tensor), dtype=torch.long)
            time_feature_tensor = torch.zeros(len(data_tensor), 8, dtype=torch.float)
            dataset = TensorDataset(data_tensor, label_tensor, location_tensor, time_feature_tensor)
        
        if self.is_train:
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
        else:
            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        return loader


class CAS_unlabel_Dataset(object):
    def __init__(self, data_config, data_csv, fold):
        """
        :param data_config: configuration module
        :param data_csv: metadata dataframe
        :param fold: "dev"/"eval" for development/evaluation set
        """
        self.stats_csv = data_csv
        if fold == "dev":
            self.root_path = data_config.dev_fea_root_path
        else:
            self.root_path = data_config.eval_fea_root_path
        self.tar_sr = data_config.sample_rate
        self.batch_size = data_config.batch_size
        self.clip_frames = data_config.clip_frames
        self.class_2_index = data_config.class_2_index
        self.enable_multimodal = getattr(data_config, 'enable_multimodal', False)
        self.location_2_index = getattr(data_config, 'location_2_index', {})

        self.file_list = []
        self.label_list = []
        self.location_list = []
        self.time_feature_list = []
        self.get_file_list()

    def get_file_list(self):
        for index, row in self.stats_csv.iterrows():
            label_str = row.get("scene_label", np.nan)

            if not isinstance(label_str, str):  
                filename = row["filename"]
                file_path = os.path.join(self.root_path, filename + ".npy")
                self.file_list.append(file_path)
                self.label_list.append(label_str)
                
                if self.enable_multimodal:
                    # extract city info
                    location = row.get("location", "unknown")
                    location_idx = self.location_2_index.get(location, 0)
                    self.location_list.append(location_idx)
                    
                    # # extract time info
                    record_time = row.get("record_time", "")
                    time_features = extract_time_features(record_time)
                    self.time_feature_list.append(time_features)

    def get_numpy_dataset(self):
        data = []
        label = []
        locations = []
        time_features = []
        
        for file_path in tqdm(self.file_list):
            file = np.load(file_path, allow_pickle=True)
            file = file[: self.clip_frames, :]
            data.append(file)

        for label_str in tqdm(self.label_list):
            label.append(label_str)

        data = np.asarray(data)
        label = np.asarray(label)
        
        if self.enable_multimodal:
            locations = np.asarray(self.location_list, dtype=np.int32)
            time_features = np.asarray(self.time_feature_list, dtype=np.float32)

        return data, label, locations, time_features

    def get_tensordataset(self):
        data, label, locations, time_features = self.get_numpy_dataset()
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(label).float()
        
        if self.enable_multimodal and len(locations) > 0:
            location_tensor = torch.from_numpy(locations).long()
            time_feature_tensor = torch.from_numpy(time_features).float()
            dataset = TensorDataset(data_tensor, label_tensor, location_tensor, time_feature_tensor)
        else:
            location_tensor = torch.zeros(len(data_tensor), dtype=torch.long)
            time_feature_tensor = torch.zeros(len(data_tensor), 8, dtype=torch.float)
            dataset = TensorDataset(data_tensor, label_tensor, location_tensor, time_feature_tensor)

        loader = DataLoader(
            dataset=dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        return loader
