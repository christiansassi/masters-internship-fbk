# Copyright (c) 2021 @ FBK - Fondazione Bruno Kessler
# Author: Maged Abdelaty
# Project: DAICS: A Deep Learning Solution for Anomaly Detection in Industrial Control Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# imports and dataset hyperparameters

import os
import argparse
import random
import scipy.signal
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder
torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from os import chdir
from os.path import realpath, dirname

chdir(dirname(realpath(__file__)))

# os.makedirs("datasets", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
# Path of the dataset files
data_path = "../datasets/SWaT2015/original"
# Path of the pretrained models
checkpoints_path = "checkpoints/"

import pickle

def swat_execute(mode='train'):
    # Start and end of 36 attacks in the SWaT dataset
    global attack_indices
    attack_indices = [(1754, 2693), (3068, 3510), (4920, 5302), (6459, 6848),
                    (7255, 7450), (7705, 8133), (11410, 12373), (15380, 15540),
                    (15541, 16100), (90685, 90917), (92140, 92570), (93445, 93720),
                    (103092, 103808), (115843, 116101), (116143, 116537),
                    (117000, 117720), (132918, 133380), (142954, 143650),
                    (172268, 172588), (172910, 173521), (198296, 199740),
                    (227828, 229518), (229519, 263727), (279120, 279240),
                    (280060, 281230), (302653, 303019), (347679, 348279),
                    (361191, 361634), (371479, 371579), (371855, 372335),
                    (389680, 390219), (436541, 437009), (437417, 437697),
                    (438147, 438547), (438621, 438917), (443501, 445190)]

    # Names of sensors and actuators in the dataset (i.e., columns of the dataset)
    global ACTUATORS_SENSORS
    ACTUATORS_SENSORS = ['FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202',
                        'AIT203', 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205',
                        'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303',
                        'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401',
                        'P401', 'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503',
                        'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502',
                        'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603']
    # List of sensors in the dataset
    global FEATURES_OUT_SEN_FLAT
    FEATURES_OUT_SEN_FLAT = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'DPIT301',
                            'FIT301', 'LIT301', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'AIT501',
                            'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504',
                            'PIT501', 'PIT502', 'PIT503', 'FIT601'] 
    # List of actuators in the dataset
    global FEATURES_OUT_ACT_FLAT
    FEATURES_OUT_ACT_FLAT = ['MV101', 'P101', 'P102', 'MV201', 'P201', 'P202', 'P203', 'P204',
                            'P205', 'P206', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302',
                            'P401', 'P402', 'P403', 'P404', 'UV401', 'P501', 'P502', 'P601',
                            'P602', 'P603']          
    # List of sensors attached to the first PLC 
    global FEATURES_OUT_SEN_1
    FEATURES_OUT_SEN_1 = ['FIT101', 'LIT101']
    # List of sensors attached to the second PLC
    global FEATURES_OUT_SEN_2
    FEATURES_OUT_SEN_2 = ['AIT201', 'AIT202', 'AIT203', 'FIT201']
    # List of sensors attached to the third PLC
    global FEATURES_OUT_SEN_3
    FEATURES_OUT_SEN_3 = ['DPIT301', 'FIT301', 'LIT301']
    # List of sensors attached to the fourth PLC
    global FEATURES_OUT_SEN_4
    FEATURES_OUT_SEN_4 = ['AIT401', 'AIT402', 'FIT401', 'LIT401']
    # List of sensors attached to the fifth PLC
    global FEATURES_OUT_SEN_5
    FEATURES_OUT_SEN_5 = ['AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 
                        'FIT503', 'FIT504', 'PIT501', 'PIT502', 'PIT503']
    # List of sensors attached to the sixth PLC
    global FEATURES_OUT_SEN_6
    FEATURES_OUT_SEN_6 = ['FIT601']                                             
    # List of lists. It is used to loop over the sensors attached to each PLC / neural network output section
    global FEATURES_OUT_SEN
    FEATURES_OUT_SEN = [FEATURES_OUT_SEN_1, FEATURES_OUT_SEN_2, FEATURES_OUT_SEN_3, FEATURES_OUT_SEN_4, FEATURES_OUT_SEN_5, FEATURES_OUT_SEN_6]

    # Number of input features to the neural network.  
    # Note that the neural network takes as input the readings of both sensors and actuators
    global FEATURES_IN
    FEATURES_IN = len(ACTUATORS_SENSORS)

    # Batch size for training and also for decision rate.
    # The ADS will produce a decision every 32 samples
    global BATCH_SIZE
    BATCH_SIZE = 32
    # Hyperparameters of the training SGD optimizer 
    global LEARNING_RATE
    LEARNING_RATE = 0.01  # ReduceLROnPlateau
    global MOMENTUM
    MOMENTUM = 0.9
    global WEIGHT_DECAY
    WEIGHT_DECAY = 1e-6
    # Length of the prediction window of the wide and deep neural network (W_{out} in the paper)
    global WINDOW_PRESENT
    WINDOW_PRESENT = 4 
    # Length of the input window of the wide and deep neural network (W_{in} in the paper)
    global WINDOW_PAST
    WINDOW_PAST = 60 
    # Time between W_{in} and W_{out} (H in the paper)
    global HORIZON
    HORIZON = 50
    # Number of training epochs of the wide and deep neural network
    global EPOCHS
    EPOCHS = 50
    # Number of training epochs of the threshold neural networks
    global EPOCHS_THRESHOLD
    EPOCHS_THRESHOLD = 5
    # Hyperperameter of the ReduceLROnPlateau torch function
    global PATIENCE
    PATIENCE = 10
    # Kernel size of the convolutional layers of the neural networks
    global KERNEL_SIZE
    KERNEL_SIZE = 2
    # Report an anomaly in a sensor if the error exceeds the threshold for W_ANOMALY seconds
    # The goal is to avoid alarms for transient changes in behaviour of sensors
    global W_ANOMALY
    W_ANOMALY = 30
    # Report alarms after this grace time (helps to ignore short alarms)
    # Zero means report all alarms
    global W_GRACE
    W_GRACE = 0 
    # Kernel length of the median filter used to smooth the prediction error
    global MED_FILTER_LAG
    MED_FILTER_LAG = 59 
    # Number of tuning epochs after a human intervention
    global T_EPOCHS
    T_EPOCHS = 10   
    # Threshold tuning learning rate
    global THRESHOLD_LR
    THRESHOLD_LR = 0.01 
    # Output section tuning learning rate
    global OUTPUT_LR
    OUTPUT_LR = 0.01
    # Start of the first W_{out} in the three datasets
    global SAMPLING_START
    SAMPLING_START = WINDOW_PAST + HORIZON + WINDOW_PRESENT
    # Step of the sliding window in the training set
    global TRAINING_STEP
    TRAINING_STEP = 1
    # Step of the sliding window in the validation set
    global VAL_STEP
    VAL_STEP = 1
    # Step of the sliding window in the test set
    global TEST_STEP
    TEST_STEP = 1

    # Name of the dataset
    global FILE_STARTER
    FILE_STARTER = "swat"
    # Path of the best parameters of the wide and deep neural network
    global best_model
    best_model = checkpoints_path + "model_" + FILE_STARTER + ".pt"
    # Path of the best parameters of the threshold neural network
    global best_pred_error_model_path
    best_pred_error_model_path = checkpoints_path + "pred_error_model_" + FILE_STARTER + ".pt"


    if mode == 'train':
        df_train, df_val, df_test, all_labels = read_swat_dataset()
        idx_train_wt, idx_train_wt_1, idx_val_wt, idx_val_wt_1, idx_test_wt, idx_test_wt_1 = prepare_sliding_windows(df_train, df_val, df_test)
        best_model = train_wide_deep_network(df_train, df_val, idx_train_wt, idx_train_wt_1, idx_val_wt, idx_val_wt_1)
        best_pred_error_model_path = train_threshold_nn(df_val, idx_val_wt, idx_val_wt_1, best_model)
    if mode == 'eval':
        df_train, df_val, df_test, all_labels = read_swat_dataset()
        idx_train_wt, idx_train_wt_1, idx_val_wt, idx_val_wt_1, idx_test_wt, idx_test_wt_1 = prepare_sliding_windows(df_train, df_val, df_test)
        thresh_sen_base = clac_threshold_base(df_val, idx_val_wt, idx_val_wt_1)
        test(df_train, df_val, df_test, all_labels, idx_test_wt, idx_test_wt_1, thresh_sen_base)



def wadi_execute(mode='train'):
    # Start and end of 15 attacks in the WADI dataset
    global attack_indices
    attack_indices = [(5103, 6617),
                      (59053, 59643),
                      (60903, 62643),
                      (63043, 63893),
                      (70773, 71443),
                      (74900, 75598),
                      (78627, 83088),
                      (85203, 85783),
                      (147303, 147390),
                      (148677, 149483),
                      (149792, 150423),
                      (151143, 151503),
                      (151651, 151853),
                      (152163, 152739),
                      (163593, 164223)]

    # Names of sensors and actuators in the dataset (i.e., columns of the dataset) 
    global ACTUATORS_SENSORS
    ACTUATORS_SENSORS = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV', '1_FIT_001_PV', '1_LS_001_AL', 
                        '1_LS_002_AL', '1_LT_001_PV', '1_MV_001_STATUS', '1_MV_002_STATUS','1_MV_003_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', 
                        '1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS', '1_P_005_STATUS', '1_P_006_STATUS', '2_MV_001_STATUS', 
                        '2_MV_002_STATUS', '2_MV_003_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_006_STATUS', '2_LT_001_PV', '2_LT_002_PV',
                        '2_FIT_001_PV', '2_PIT_001_PV', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV', '2_FIT_002_PV', 
                        '2_FIT_003_PV', '2_PIT_002_PV','2_PIT_003_PV', '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV', 
                        '2_MV_009_STATUS', '2_P_003_SPEED', '2_P_003_STATUS', '2_P_004_SPEED', '2_P_004_STATUS', '2_MCV_007_CO', '2_PIC_003_CO', 
                        '2_PIC_003_PV', '2_PIC_003_SP', '2_DPIT_001_PV', '2_FIC_101_CO','2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', 
                        '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV', '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', 
                        '2_FIC_401_SP', '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', '2_FIC_601_PV', '2_FIC_601_SP', 
                        '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', '2_FQ_501_PV', '2_FQ_601_PV', '2_LS_101_AH', '2_LS_101_AL', 
                        '2_LS_201_AH', '2_LS_201_AL', '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', '2_LS_501_AL', 
                        '2_LS_601_AH', '2_LS_601_AL', '2_MCV_101_CO', '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', 
                        '2_MCV_601_CO', '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', '2_MV_501_STATUS', 
                        '2_MV_601_STATUS', '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS', 
                        '2_SV_601_STATUS', '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV', '3_FIT_001_PV', 
                        '3_LS_001_AL', '3_LT_001_PV', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS', '3_P_002_STATUS',
                        '3_P_003_STATUS', '3_P_004_STATUS', 'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW', 'PLANT_START_STOP_LOG'] # 123 sensors and actuators
    # List of sensors in the dataset
    global FEATURES_OUT_SEN_FLAT
    FEATURES_OUT_SEN_FLAT = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV', '1_FIT_001_PV', '1_LT_001_PV',
               '2_DPIT_001_PV', '2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP',
               '2_FIC_301_CO', '2_FIC_301_PV', '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', '2_FIC_501_CO',
               '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV',
               '2_FIT_003_PV', '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', '2_FQ_501_PV', '2_FQ_601_PV', 
               '2_LT_001_PV', '2_LT_002_PV', '2_MCV_007_CO', '2_MCV_101_CO', '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO',
               '2_MCV_501_CO', '2_MCV_601_CO', '2_P_003_SPEED', '2_P_004_SPEED', '2_PIC_003_PV', '2_PIC_003_SP', '2_PIC_003_CO', '2_PIT_001_PV',
               '2_PIT_002_PV', '2_PIT_003_PV', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV', '2B_AIT_001_PV',
               '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV', '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV',
               '3_AIT_005_PV', '3_FIT_001_PV', '3_LT_001_PV', 'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW'] # 69 sensors
    # List of actuators in the dataset
    global FEATURES_OUT_ACT_FLAT
    FEATURES_OUT_ACT_FLAT = ['1_LS_001_AL' ,'1_LS_002_AL' ,'1_MV_001_STATUS' ,'1_MV_002_STATUS' ,'1_MV_003_STATUS' ,'1_MV_004_STATUS' ,
                        '1_P_001_STATUS' ,'1_P_002_STATUS' ,'1_P_003_STATUS' ,'1_P_004_STATUS' ,'1_P_005_STATUS' ,'1_P_006_STATUS' , '2_MV_001_STATUS',
                        '2_MV_002_STATUS', '2_MV_003_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_006_STATUS', '2_MV_009_STATUS',
                        '2_P_003_STATUS' , '2_P_004_STATUS', '2_LS_101_AH' ,'2_LS_101_AL' ,'2_LS_201_AH' ,'2_LS_201_AL'
                        ,'2_LS_301_AH' ,'2_LS_301_AL' ,'2_LS_401_AH' ,'2_LS_401_AL' ,'2_LS_501_AH' ,'2_LS_501_AL' ,'2_LS_601_AH' ,'2_LS_601_AL',
                        '2_MV_101_STATUS' ,'2_MV_201_STATUS' ,'2_MV_301_STATUS' ,'2_MV_401_STATUS' ,'2_MV_501_STATUS'
                        ,'2_MV_601_STATUS','2_SV_101_STATUS' ,'2_SV_201_STATUS' ,'2_SV_301_STATUS' ,'2_SV_401_STATUS' ,'2_SV_501_STATUS'
                        ,'2_SV_601_STATUS' , '3_LS_001_AL' ,'3_MV_001_STATUS' ,'3_MV_002_STATUS' ,'3_MV_003_STATUS' ,'3_P_001_STATUS'
                        ,'3_P_002_STATUS' ,'3_P_003_STATUS' ,'3_P_004_STATUS' ,'PLANT_START_STOP_LOG']  # 54 actuators

    # List of sensors attached to the first PLC
    global FEATURES_OUT_SEN_1
    FEATURES_OUT_SEN_1 = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', '1_AIT_005_PV', '1_FIT_001_PV', '1_LT_001_PV'] # sensors plc 1
    # List of sensors attached to the second PLC
    global FEATURES_OUT_SEN_2
    FEATURES_OUT_SEN_2 = ['2_LT_001_PV', '2_LT_002_PV', '2_FIT_001_PV', '2_PIT_001_PV', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV'] # sensors plc 2
    # List of sensors attached to the third PLC
    global FEATURES_OUT_SEN_3
    FEATURES_OUT_SEN_3 = ['2_P_003_SPEED', '2_P_004_SPEED', '2_MCV_007_CO', '2_FIT_002_PV', '2_FIT_003_PV', '2_PIT_002_PV','2_PIT_003_PV', 
                            '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV', '2_PIC_003_PV', '2_PIC_003_SP', '2_PIC_003_CO'] # sensors plc 3
    # List of sensors attached to the fourth PLC
    global FEATURES_OUT_SEN_4
    FEATURES_OUT_SEN_4 = ['2_DPIT_001_PV', '2_FIC_101_CO','2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP', 
                            '2_FIC_301_CO', '2_FIC_301_PV', '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', '2_FIC_501_CO', 
                            '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', '2_FIC_601_PV', '2_FIC_601_SP', '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', 
                            '2_FQ_401_PV', '2_FQ_501_PV', '2_FQ_601_PV', '2_LS_101_AH', '2_MCV_101_CO', '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', 
                            '2_MCV_501_CO', '2_MCV_601_CO'] # sensors plc 4
    # List of sensors attached to the fifth PLC
    global FEATURES_OUT_SEN_5
    FEATURES_OUT_SEN_5 = ['3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV', '3_FIT_001_PV', '3_LT_001_PV', 
                            'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW'] # sensors plc 5

    # List of lists. It is used to loop over the sensors attached to each PLC / neural network output section
    global FEATURES_OUT_SEN
    FEATURES_OUT_SEN = [FEATURES_OUT_SEN_1, FEATURES_OUT_SEN_2, FEATURES_OUT_SEN_3, FEATURES_OUT_SEN_4, FEATURES_OUT_SEN_5]

    # Number of input features to the neural network.  
    # Note that the neural network takes as input the readings of both sensors and actuators
    global FEATURES_IN
    FEATURES_IN = len(ACTUATORS_SENSORS)

    # Batch size for training and also for decision rate.
    # The ADS will produce a decision every 32 samples
    global BATCH_SIZE
    BATCH_SIZE = 32
    # Hyperparameters of the training SGD optimizer 
    global LEARNING_RATE
    LEARNING_RATE = 0.01  # ReduceLROnPlateau
    global MOMENTUM
    MOMENTUM = 0.9
    global WEIGHT_DECAY
    WEIGHT_DECAY = 1e-6
    # Length of the prediction window of the wide and deep neural network (W_{out} in the paper)
    global WINDOW_PRESENT
    WINDOW_PRESENT = 4  
    # Length of the input window of the wide and deep neural network (W_{in} in the paper)
    global WINDOW_PAST
    WINDOW_PAST = 50 
    # Time between W_{in} and W_{out} (H in the paper)
    global HORIZON
    HORIZON = 20
    # Number of training epochs of the wide and deep neural network
    global EPOCHS
    EPOCHS = 50
    # Number of training epochs of the threshold neural networks
    global EPOCHS_THRESHOLD
    EPOCHS_THRESHOLD = 5
    # Hyperperameter of the ReduceLROnPlateau torch function
    global PATIENCE
    PATIENCE = 40
    # Kernel size of the convolutional layers of the neural networks
    global KERNEL_SIZE
    KERNEL_SIZE = 5
    # Report an anomaly in a sensor if the error exceeds the threshold for W_ANOMALY seconds
    # The goal is to avoid alarms for transient changes in behaviour of sensors
    global W_ANOMALY
    W_ANOMALY = 30
    # Report alarms after this grace time (helps to ignore short alarms)
    # Zero means report all alarms
    global W_GRACE
    W_GRACE = 0 
    # Kernel length of the median filter used to smooth the prediction error
    global MED_FILTER_LAG
    MED_FILTER_LAG = 59 
    # Number of tuning epochs after a human intervention
    global T_EPOCHS
    T_EPOCHS = 100   
    # Threshold tuning learning rate
    global THRESHOLD_LR
    THRESHOLD_LR = 0.01 
    # Output section tuning learning rate
    global OUTPUT_LR
    OUTPUT_LR = 0.01
    # Start of the first W_{out} in the three datasets
    global SAMPLING_START
    SAMPLING_START = WINDOW_PAST + HORIZON + WINDOW_PRESENT
    # Step of the sliding window in the training set
    global TRAINING_STEP
    TRAINING_STEP = 1
    # Step of the sliding window in the validation set
    global VAL_STEP
    VAL_STEP = 1
    # Step of the sliding window in the test set
    global TEST_STEP
    TEST_STEP = 1


    # Name of the dataset
    global FILE_STARTER
    FILE_STARTER = "wadi"
    # Path of the best parameters of the wide and deep neural network
    global best_model
    best_model = checkpoints_path + "model_" + FILE_STARTER + ".pt"
    # Path of the best parameters of the threshold neural network
    global best_pred_error_model_path 
    best_pred_error_model_path = checkpoints_path + "pred_error_model_" + FILE_STARTER + ".pt"



    if mode == 'train':
        df_train, df_val, df_test, all_labels = read_wadi_dataset()
        idx_train_wt, idx_train_wt_1, idx_val_wt, idx_val_wt_1, idx_test_wt, idx_test_wt_1 = prepare_sliding_windows(df_train, df_val, df_test)
        best_model = train_wide_deep_network(df_train, df_val, idx_train_wt, idx_train_wt_1, idx_val_wt, idx_val_wt_1)
        best_pred_error_model_path = train_threshold_nn(df_val, idx_val_wt, idx_val_wt_1, best_model)
    if mode == 'eval':
        df_train, df_val, df_test, all_labels = read_wadi_dataset()
        idx_train_wt, idx_train_wt_1, idx_val_wt, idx_val_wt_1, idx_test_wt, idx_test_wt_1 = prepare_sliding_windows(df_train, df_val, df_test)
        thresh_sen_base = clac_threshold_base(df_val, idx_val_wt, idx_val_wt_1)
        test(df_train, df_val, df_test, all_labels, idx_test_wt, idx_test_wt_1, thresh_sen_base)



# Fix seeds
# source: https://github.com/pytorch/pytorch/issues/7068
def seed_torch(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Prepare the dataset

def read_swat_dataset():
    df_train = pd.read_csv(os.path.join(data_path, "SWaT_Dataset_Normal.csv"), header=[0])
    # Remove spaces from columns names
    df_train.columns=df_train.columns.str.strip()
    # Keep data of sensors and actuators
    df_train = df_train[ACTUATORS_SENSORS].astype(float)
    # drop rows with null values
    df_train.dropna(inplace=True)

    df_test = pd.read_csv(os.path.join(data_path, "SWaT_Dataset_Attack.csv"), header=[0])
    df_test.columns=df_test.columns.str.strip()
    df_test = df_test[ACTUATORS_SENSORS].astype(float)
    df_test.dropna(inplace=True)
    # Prepare labels of the test dataset
    # the training dataset contains only normal records
    all_labels = np.zeros(len(df_test))
    for att in attack_indices:
        all_labels[att[0]: att[1]] = 1
    # Split part of the training dataset for validation
    train_val_split = int(0.8 * df_train.shape[0])
    df_val = df_train[train_val_split:]
    df_train = df_train[:train_val_split]
    # Find minimum and maximum values of each feature
    data = np.concatenate([df_train.values, df_val.values, df_test.values], axis=0)
    min_v = data.min(axis=0)
    max_v = data.max(axis=0)
    # The goal of the next two lines is to avoid dividing by zero when normalising the features between zero and one
    min_v = np.where(min_v == max_v, np.zeros_like(min_v), min_v)
    max_v = np.where(max_v == 0., np.ones_like(max_v), max_v)
    data = None
    # Normalise the training set between zero and one
    df_train = (df_train - min_v) / (max_v - min_v)
    df_train.clip(lower=0, upper=1, inplace=True)
    # Normalise the validation set between zero and one
    df_val = (df_val - min_v) / (max_v - min_v)
    df_val.clip(lower=0, upper=1, inplace=True)
    # Normalise the test set between zero and one
    df_test = (df_test - min_v) / (max_v - min_v)
    df_test.clip(lower=0, upper=1, inplace=True)
    return df_train, df_val, df_test, all_labels



def read_wadi_dataset():
    df_train = pd.read_csv(os.path.join(data_path,"WADI_14days_new.csv"), delimiter=',', dtype="str", header=[0])
    # Keep data of sensors and actuators
    df_train = df_train[ACTUATORS_SENSORS].astype(float)
    # drop rows with null values
    df_train.dropna(inplace=True)    

    df_test = pd.read_csv(os.path.join(data_path,"WADI_attackdataLABLE.csv"), delimiter=',', skiprows=[0], dtype="str", header=[0])
    df_test = df_test[ACTUATORS_SENSORS].astype(float)
    df_test.dropna(inplace=True)
    # Prepare labels of the test dataset
    # the training dataset contains only normal records
    all_labels = np.zeros(len(df_test))
    for att in attack_indices:
        all_labels[att[0]: att[1]] = 1
    # Split part of the training dataset for validation
    train_val_split = int(0.95 * df_train.shape[0])
    df_val = df_train[train_val_split:]
    df_train = df_train[:train_val_split]
    # Find minimum and maximum values of each feature
    data = np.concatenate([df_train.values, df_val.values, df_test.values], axis=0)
    min_v = data.min(axis=0)
    max_v = data.max(axis=0)
    # The goal of the next two lines is to avoid dividing by zero when normalising the features between zero and one
    min_v = np.where(min_v == max_v, np.zeros_like(min_v), min_v)
    max_v = np.where(max_v == 0., np.ones_like(max_v), max_v)
    data = None
    # Normalise the training set between zero and one
    df_train = (df_train - min_v) / (max_v - min_v)
    df_train.clip(lower=0, upper=1, inplace=True)
    # Normalise the validation set between zero and one
    df_val = (df_val - min_v) / (max_v - min_v)
    df_val.clip(lower=0, upper=1, inplace=True)
    # Normalise the test set between zero and one
    df_test = (df_test - min_v) / (max_v - min_v)
    df_test.clip(lower=0, upper=1, inplace=True)
    return df_train, df_val, df_test, all_labels





def prepare_sliding_windows(df_train, df_val, df_test):
    
    # Indices of W_{out} 
    idx_train_wt = np.arange(SAMPLING_START, len(df_train), TRAINING_STEP)[:, None] - np.arange(1, WINDOW_PRESENT + 1)
    idx_train_wt = np.sort(idx_train_wt)
    idx_train_wt = idx_train_wt[: (len(idx_train_wt) // BATCH_SIZE) * BATCH_SIZE, :]
    # Indices of W_{in}
    idx_train_wt_1 = (np.arange(SAMPLING_START, len(df_train), TRAINING_STEP) - HORIZON - WINDOW_PRESENT)[:, None] - np.arange(1, WINDOW_PAST + 1)
    idx_train_wt_1 = np.sort(idx_train_wt_1)
    idx_train_wt_1 = idx_train_wt_1[: (len(idx_train_wt_1) // BATCH_SIZE) * BATCH_SIZE, :]
    print("Number of batches in the train dataset", len(idx_train_wt) // BATCH_SIZE)
    # Indices of W_{out} 
    idx_val_wt = np.arange(SAMPLING_START, len(df_val), VAL_STEP)[:, None] - np.arange(1, WINDOW_PRESENT + 1)
    idx_val_wt = np.sort(idx_val_wt)
    idx_val_wt = idx_val_wt[: (len(idx_val_wt) // BATCH_SIZE) * BATCH_SIZE, :]
    # Indices of W_{in}
    idx_val_wt_1 = (np.arange(SAMPLING_START, len(df_val), VAL_STEP) - HORIZON - WINDOW_PRESENT)[:, None] - np.arange(1, WINDOW_PAST + 1)
    idx_val_wt_1 = np.sort(idx_val_wt_1)
    idx_val_wt_1 = idx_val_wt_1[: (len(idx_val_wt_1) // BATCH_SIZE) * BATCH_SIZE, :]
    print("Number of batches in the val dataset", len(idx_val_wt) // BATCH_SIZE)
    # Indices of W_{out}
    idx_test_wt = np.arange(SAMPLING_START, len(df_test), TEST_STEP)[:, None] - np.arange(1, WINDOW_PRESENT + 1)
    idx_test_wt = np.sort(idx_test_wt)
    idx_test_wt = idx_test_wt[: (len(idx_test_wt) // BATCH_SIZE) * BATCH_SIZE, :]
    # Indices of W_{in}
    idx_test_wt_1 = (np.arange(SAMPLING_START, len(df_test), TEST_STEP) - HORIZON - WINDOW_PRESENT)[:, None] - np.arange(1, WINDOW_PAST + 1)
    idx_test_wt_1 = np.sort(idx_test_wt_1)
    idx_test_wt_1 = idx_test_wt_1[: (len(idx_test_wt_1) // BATCH_SIZE) * BATCH_SIZE, :]
    print("Number of batches in the test dataset", len(idx_test_wt) // BATCH_SIZE)

    return idx_train_wt, idx_train_wt_1, idx_val_wt, idx_val_wt_1, idx_test_wt, idx_test_wt_1

# Neural networks

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def conv1d_output_shape(l_in, kernel_size=1, stride=1, pad=0, dilation=1):
    l_out = np.floor((l_in + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1)
    return int(l_out)


def maxpool1d_output_shape(l_in, kernel_size=1, stride=None, pad=0, dilation=1):
    if stride is None:
        stride = kernel_size
    l_out = np.floor((l_in + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1)
    return int(l_out if l_out != 0 else 1)


class ModelFExtractor(nn.Module):
    def __init__(self, window_size_in, window_size_out, n_devices_in, kernel_size):
        super(ModelFExtractor, self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.4)

        self.fc13 = nn.Linear(window_size_in, window_size_in * 3 if window_size_in >= n_devices_in else n_devices_in * 3)
        self.conv = nn.Sequential(
            nn.Conv1d(n_devices_in, 64, kernel_size),
            nn.LeakyReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size),
            nn.LeakyReLU(True),
            nn.MaxPool1d(2))
        self.conv_out_channels = 128
        self.maxpool1_out = maxpool1d_output_shape(conv1d_output_shape(window_size_in * 3 if window_size_in >= n_devices_in else n_devices_in * 3, kernel_size=kernel_size), kernel_size=2)
        self.maxpool2_out = maxpool1d_output_shape(conv1d_output_shape(self.maxpool1_out, kernel_size=kernel_size), kernel_size=2)
        self.out_2 = nn.Linear(self.conv_out_channels, window_size_out)

        self.fc20 = nn.Linear(window_size_in, window_size_out)

        self.out_h = nn.Linear(self.maxpool2_out + n_devices_in, 80)

    def forward_two(self, x):
        # Convolutional branch
        x = self.fc13(x)
        x = self.conv(x)
        x = x.view(x.size(0), self.maxpool2_out, self.conv_out_channels)
        x = self.dropout(x)
        x = self.relu(self.out_2(x))
        return x

    def forward_three(self, x):
        # Wide branch
        x = self.dropout(self.relu(self.fc20(x)))
        return x

    def forward(self, x_t_1):
        x_t_1 = x_t_1.transpose(2, 1)
        y_t2 = self.forward_two(x_t_1)
        y_t3 = self.forward_three(x_t_1)
        y_t = torch.cat((y_t2, y_t3), dim=1).transpose(2, 1)
        y_t = self.dropout(self.relu(self.out_h(y_t)))
        return y_t


class ModelSensors(nn.Module):
    def __init__(self, n_devices_out):
        super(ModelSensors, self).__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.out_h1 = nn.Linear(80, int(n_devices_out * 2.25))
        self.out_h2 = nn.Linear(int(n_devices_out * 2.25), int(n_devices_out * 1.5))
        self.out = nn.Linear(int(n_devices_out * 1.5), n_devices_out)

    def forward(self, y_t):
        y_t = self.dropout(self.relu(self.out_h1(y_t)))
        y_t = self.dropout(self.relu(self.out_h2(y_t)))
        y_t = self.relu(self.out(y_t))
        return y_t

# Threshold prediction neural network

class PredErrorModel(nn.Module):

    def __init__(self, window_size_in, window_size_out):
        super(PredErrorModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 2, 2),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(2, 4, 2),
            nn.ReLU(True),
            nn.MaxPool1d(2),
        )
        self.maxpool1_out = maxpool1d_output_shape(conv1d_output_shape(window_size_in, kernel_size=2),
                                                   kernel_size=2)
        self.maxpool2_out = maxpool1d_output_shape(conv1d_output_shape(self.maxpool1_out, kernel_size=2),
                                                   kernel_size=2)
        self.out_1 = nn.Linear(self.maxpool2_out, 1)

    def forward(self, x_t_1):
        x = x_t_1.transpose(2, 1)
        x = self.conv(x)
        x = self.relu(self.out_1(x))
        return x

# loss functions

criterion_train = nn.MSELoss()


def train_loss_function(recon_x, x):
    assert recon_x.size() == x.size()
    return criterion_train(recon_x, x)


criterion_test = nn.MSELoss(reduction='none')


def test_loss_function(recon_x, x):
    assert recon_x.size() == x.size()
    return torch.mean(criterion_test(recon_x, x), dim=2)


criterion_threshold_test = nn.MSELoss()


def threshold_loss_function(recon_x, x):
    assert recon_x.size() == x.size()
    return criterion_threshold_test(recon_x, x)


criterion_ftune = nn.MSELoss()


def ftune_loss_function(recon_x, x):
    assert recon_x.size() == x.size()
    return criterion_ftune(recon_x, x)

# Training

def train_wide_deep_network(df_train, df_val, idx_train_wt, idx_train_wt_1, idx_val_wt, idx_val_wt_1):

    # Create the feature extractor part of the wide and deep neural network
    model_f_extractor = ModelFExtractor(window_size_in=WINDOW_PAST, window_size_out=WINDOW_PRESENT, n_devices_in=FEATURES_IN, kernel_size=KERNEL_SIZE)
    # Move the model to the GPU if available
    model_f_extractor.to(device)
    # Prepare list of parameters of the feature extractor to be given to the SGD optimizer
    params = list(model_f_extractor.parameters())
    # List of output models. Each one models the normal behaviour of a group of sensors
    model_sensors = []
    for feat_sen in FEATURES_OUT_SEN:
        mdl = ModelSensors(n_devices_out=len(feat_sen))
        mdl.to(device)
        params += list(mdl.parameters())
        model_sensors.append(mdl)

    # Give the parameters of the wide and deep neural network to the optimizer
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9)
    # Reduce the learning rate after it stops decreasing for PATIENCE epochs
    scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE)
    # Print summary of models
    print_summary = False
    train_loss = list()
    val_loss = list()
    # Best validation score
    val_loss_min = None

    for epoch in range(EPOCHS):

        print("training", len(idx_train_wt) / BATCH_SIZE, "batches")
        print("epoch", epoch + 1, "out of", EPOCHS)
        # Prepare models for training
        model_f_extractor.eval()
        for mdl in model_sensors:
            mdl.eval()


        before = time.time()
        trn_loss = 0.0
        # Loop over the training set. Permutate the training set on each epoch
        for batch_idx in np.random.permutation(range(0, len(idx_train_wt) // BATCH_SIZE)):
            # List of W_{out} of each output section in the wide and deep neural network
            window_t_sen = []
            for feat_sen in FEATURES_OUT_SEN:
                # Note: feat_sen is the sensors assigned to an output section in the neural network
                win_t_sen = df_train.iloc[idx_train_wt[batch_idx * BATCH_SIZE:  batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].loc[:, feat_sen].values.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
                # Move each W_{out} to the GPU
                win_t_sen = torch.from_numpy(win_t_sen).float().to(device)
                window_t_sen.append(win_t_sen)
            # Prepare W_{in}
            window_t_1 = df_train.iloc[idx_train_wt_1[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].values.reshape(BATCH_SIZE, WINDOW_PAST, -1)
            # Move W_{in} to the GPU
            window_t_1 = torch.from_numpy(window_t_1).float().to(device)

            if print_summary:
                print(summary(model_f_extractor, (window_t_1.size()[1:])))
                for mdl in model_sensors:
                    print(summary(mdl, (4, 80)))
                print_summary = False
            # Set the model gradients to zero
            model_f_extractor.zero_grad()
            loss = 0
            # Calculate output of layer DL4 (last layer in the feature extractor)
            pickle.dump(window_t_1, open("w_in_daics.pkl", "wb+"))

            y_t = model_f_extractor(window_t_1.float().to(device))

            pickle.dump(y_t, open(f"w_extra_daics.pkl", "wb+"))

            for mdl_ix, mdl in enumerate(model_sensors):
                mdl.zero_grad()
                # Calculate output of each output section (prediction of normal behaviour)
                y_t_1 = mdl(y_t)
                # Calculate the training loss of the current batch
                loss += train_loss_function(y_t_1, window_t_sen[mdl_ix])

                pickle.dump(window_t_sen[mdl_ix], open(f"w_out{mdl_ix+1}_daics.pkl", "wb+"))
                pickle.dump(y_t_1, open(f"w_pred{mdl_ix+1}_daics.pkl", "wb+"))

            # One SGD step
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()

        print("Time elapsed on the dataset", time.time() - before)

        train_loss.append(trn_loss / (len(idx_train_wt) // BATCH_SIZE))
        print("epoch #", epoch + 1, "training loss", train_loss[-1])
        # Prepare model for evaluation
        model_f_extractor.eval()
        for mdl in model_sensors:
            mdl.eval()

        vld_loss = 0.0
        # The code is the same as before except using the validation dataset
        for batch_idx in range(0, len(idx_val_wt) // BATCH_SIZE):
            window_t_sen = []
            for feat_sen in FEATURES_OUT_SEN:
                win_t_sen = df_val.iloc[idx_val_wt[batch_idx * BATCH_SIZE:  batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].loc[:, feat_sen].values.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
                win_t_sen = torch.from_numpy(win_t_sen).float().to(device)
                window_t_sen.append(win_t_sen)
            
            window_t_1 = df_val.iloc[idx_val_wt_1[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].values.reshape(BATCH_SIZE, WINDOW_PAST, -1)
            window_t_1 = torch.from_numpy(window_t_1).float().to(device)

            loss = 0
            y_t = model_f_extractor(window_t_1.float().to(device))
            for mdl_ix, mdl in enumerate(model_sensors):
                y_t_1 = mdl(y_t)
                loss += train_loss_function(y_t_1, window_t_sen[mdl_ix])

            vld_loss += loss.item()

        val_loss.append(vld_loss / (len(idx_val_wt) // BATCH_SIZE))
        print("epoch #", epoch + 1, "validation loss", val_loss[-1])
        # Save the first checkpoint of the trained models
        if val_loss_min is None:
            val_loss_min = val_loss[-1]
            print("Model saving...epoch", epoch + 1)
            mdl_dict = {}
            mdl_dict['model_f_extractor'] = model_f_extractor.state_dict()
            for mdl_ix, mdl in enumerate(model_sensors):
                mdl_dict[mdl_ix] = mdl.state_dict()
            torch.save(mdl_dict, best_model)
        # If the validation error has decreased, save a new checkpoint
        if val_loss_min > val_loss[-1]:
            val_loss_min = val_loss[-1]
            print("Model saving...epoch", epoch + 1)
            mdl_dict['model_f_extractor'] = model_f_extractor.state_dict()
            for mdl_ix, mdl in enumerate(model_sensors):
                mdl_dict[mdl_ix] = mdl.state_dict()
            torch.save(mdl_dict, best_model)

        # Decay Learning Rate, pass validation loss for tracking at every epoch
        scheduler.step(val_loss[-1])

    print(best_model)
    return best_model

# Training threshold neural networks    
    
def train_threshold_nn(df_val, idx_val_wt, idx_val_wt_1, best_model):
    # Create the feature extractor model
    model_f_extractor = ModelFExtractor(window_size_in=WINDOW_PAST, window_size_out=WINDOW_PRESENT, n_devices_in=FEATURES_IN, kernel_size=KERNEL_SIZE)

    model_f_extractor.to(device)
    # Load the checkpoint of the best trained model
    checkpoint = torch.load(best_model, map_location=device)
    # Load the parameters from the checkpoint
    model_f_extractor.load_state_dict(checkpoint['model_f_extractor'])
    model_f_extractor.eval()

    model_sensors = []
    for ix_feat_sen, feat_sen in enumerate(FEATURES_OUT_SEN):
        mdl = ModelSensors(n_devices_out=len(feat_sen))
        mdl.to(device)
        # Load the parameters from the checkpoint
        mdl.load_state_dict(checkpoint[ix_feat_sen])
        mdl.eval()
        model_sensors.append(mdl)


    print_summary = True
    # Dictionary to store the best parameters of each threshold prediction model
    # Note: We train a model for each output section
    model_dict = {}
    for ix_mdl, mdl in enumerate(model_sensors):
        # Array to store the prediction error on the validation dataset
        all_predicted = np.zeros(len(df_val), dtype=np.float32)
        # Loop over the validation dataset to fill the prediction error array
        for batch_idx in range(0, len(idx_val_wt) // BATCH_SIZE):
            window_t= df_val.iloc[idx_val_wt[batch_idx * BATCH_SIZE:  batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].loc[:, FEATURES_OUT_SEN[ix_mdl]].values.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
            window_t_1 = df_val.iloc[idx_val_wt_1[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].values.reshape(BATCH_SIZE, WINDOW_PAST, -1)
            window_t, window_t_1 = torch.from_numpy(window_t).float().to(device), torch.from_numpy(window_t_1).float().to(device)
                
            y_t = model_f_extractor(window_t_1.float().to(device))
            y_t_1 = mdl(y_t)
            loss = test_loss_function(y_t_1, window_t)

            all_predicted[idx_val_wt[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE, 0]] =  loss.detach().cpu().numpy()[:, 0]
        # Remove any zeros at the beginning and end of the prediction error array
        all_predicted = np.trim_zeros(all_predicted)
        # Use a median filter to smooth (remove spikes) the prediction error
        all_predicted = scipy.signal.medfilt(all_predicted, kernel_size=MED_FILTER_LAG)
        # Prepare indices of W_{out} of the threshold prediction neural network
        idx_threshold_wt = np.arange(SAMPLING_START, len(all_predicted), VAL_STEP)[:, None] - np.arange(1, WINDOW_PRESENT + 1)
        idx_threshold_wt = np.sort(idx_threshold_wt)
        idx_threshold_wt = idx_threshold_wt[: (len(idx_threshold_wt) // BATCH_SIZE) * BATCH_SIZE, :]
        # Prepare indices of W_{in} of the threshold prediction neural network
        idx_threshold_wt_1 = (np.arange(SAMPLING_START, len(all_predicted), VAL_STEP) - HORIZON - WINDOW_PRESENT)[:, None] - np.arange(1, WINDOW_PAST + 1)
        idx_threshold_wt_1 = np.sort(idx_threshold_wt_1)
        idx_threshold_wt_1 = idx_threshold_wt_1[: (len(idx_threshold_wt_1) // BATCH_SIZE) * BATCH_SIZE, :]
        # Create threshold prediction neural network
        pred_error_model = PredErrorModel(window_size_in=WINDOW_PAST, window_size_out=WINDOW_PRESENT)
        # Move it to GPU
        pred_error_model.to(device)
        pred_error_optimizer = torch.optim.SGD(pred_error_model.parameters(), lr=0.01)

        
        loss_min = None
        train_loss = list()
        print_summary = True
        for epoch in range(EPOCHS_THRESHOLD):
            print("training", len(idx_threshold_wt) // BATCH_SIZE, "batches")
            print("epoch", epoch + 1, "out of", EPOCHS_THRESHOLD)
            before = time.time()
            trn_loss = 0.0
            # Loop over the prediction error array to train a threshold prediction neural network
            for batch_idx in np.random.permutation(range(0, len(idx_threshold_wt) // BATCH_SIZE)):
                window_t = all_predicted[idx_threshold_wt[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE]]
                window_t_1 = all_predicted[idx_threshold_wt_1[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE]]

                window_t, window_t_1 = torch.from_numpy(window_t[:, :, None]).float().to(device), \
                                    torch.from_numpy(window_t_1[:, :, None]).float().to(device)

                if print_summary:
                    print(summary(pred_error_model, (window_t_1.size()[1:])))
                    print_summary = False

                pred_error_model.zero_grad()
                y_t = pred_error_model(window_t_1.float().to(device)).abs()
                if np.all(y_t.detach().cpu().numpy() == 0):
                    pred_error_model.zero_grad()
                    pred_error_model.apply(init_weights)
                    y_t = pred_error_model(window_t_1.float().to(device)).abs()
                loss = train_loss_function(y_t, window_t)
                loss.backward(retain_graph=True)
                pred_error_optimizer.step()

                trn_loss += loss.detach().cpu().numpy()

            print("Time elapsed on the dataset", time.time() - before)

            train_loss.append(trn_loss / (len(idx_threshold_wt) // BATCH_SIZE))
            print("epoch #", epoch + 1, "training loss", train_loss[-1])
            # Add to the dictionary the best parameters of the trained threshold prediction neural network
            if loss_min is None:
                loss_min = train_loss[-1]
                model_dict[ix_mdl] = pred_error_model.state_dict()
            if loss_min > train_loss[-1]:
                val_loss_min = train_loss[-1]
                print("Model saving...epoch", epoch + 1)
                model_dict[ix_mdl] = pred_error_model.state_dict()
    # Save the dictionary of the best trained neural networks
    torch.save(model_dict, best_pred_error_model_path)
    print(best_pred_error_model_path)
    return best_pred_error_model_path

# Calculate threshold base T_{base}

def clac_threshold_base(df_val, idx_val_wt, idx_val_wt_1):
    # Create a wide and deep neural network
    # Load the best trained parameters
    model_sensors = []
    model_f_extractor = ModelFExtractor(window_size_in=WINDOW_PAST, window_size_out=WINDOW_PRESENT, 
                                        n_devices_in=FEATURES_IN, kernel_size=KERNEL_SIZE)
    model_f_extractor.to(device)
    model_checkpoint = torch.load(best_model, map_location=device)
    model_f_extractor.load_state_dict(model_checkpoint['model_f_extractor'])
    model_f_extractor.eval()
    # List of arrays to store the validation error of each output section
    val_loss = []

    for ix_feat_sen, feat_sen in enumerate(FEATURES_OUT_SEN):
        mdl = ModelSensors(n_devices_out=len(feat_sen))
        mdl.to(device)
        mdl.load_state_dict(model_checkpoint[ix_feat_sen])
        model_sensors.append(mdl)
        val_loss.append([])
    # Loop over the validation dataset to collect validation error
    for batch_idx in range(0, len(idx_val_wt) // BATCH_SIZE):
        window_t_sen = []
        for feat_sen in FEATURES_OUT_SEN:
            win_t_sen = df_val.iloc[idx_val_wt[batch_idx * BATCH_SIZE:  batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].loc[:, feat_sen].values.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
            win_t_sen = torch.from_numpy(win_t_sen).float().to(device)
            window_t_sen.append(win_t_sen)
        
        window_t_1 = df_val.iloc[idx_val_wt_1[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].values.reshape(BATCH_SIZE, WINDOW_PAST, -1)
        window_t_1 = torch.from_numpy(window_t_1).float().to(device)

        y_t = model_f_extractor(window_t_1.float().to(device))
        for mdl_ix, mdl in enumerate(model_sensors):
            y_t_1 = mdl(y_t)
            loss = test_loss_function(y_t_1, window_t_sen[mdl_ix])
            # Calculate and smooth the prediction error to remove spikes
            val_loss[mdl_ix].append(scipy.signal.medfilt(loss.detach().cpu().numpy()[:, 0].flatten(), kernel_size=MED_FILTER_LAG))
    # Flatten the list of arrays of validation errors
    # Calculate mean and standard deviation of the validation loss
    err_mean = np.mean([loss for loss_item in val_loss for loss in loss_item])
    err_std = np.std([loss for loss_item in val_loss for loss in loss_item])
    thresh_sen_base = []    
    print("Mean: {} , STD: {}".format(err_mean, err_std))
    print("T_base", err_mean + err_std)
    # T_{base} is the sum of mean and std of losses in the validation dataset
    for mdl_ix, mdl in enumerate(model_sensors):
        thresh_sen_base.append(err_mean + err_std)
    return thresh_sen_base

# Evaluate a trained model

def test(df_train, df_val, df_test, all_labels, idx_test_wt, idx_test_wt_1, thresh_sen_base):
    # List of threshold prediction models
    pred_error_model = []
    # List of SGD optimizers to assist with prediction the threshold
    pred_error_optimizer = []
    model_sensors = []
    # Load the best trained models
    model_checkpoint = torch.load(best_model, map_location=device)
    pred_error_chk = torch.load(best_pred_error_model_path, map_location=device)

    model_f_extractor = ModelFExtractor(window_size_in=WINDOW_PAST, window_size_out=WINDOW_PRESENT, n_devices_in=FEATURES_IN, kernel_size=KERNEL_SIZE)
    model_f_extractor.to(device)
    model_f_extractor.load_state_dict(model_checkpoint['model_f_extractor'])
    model_f_extractor.eval()

    # List of arrays to store the prediction error of each output section
    all_predicted_sen = []
    # List of arrays to store the result of anomaly detection in each output section
    all_threshold_sen = []
    # Array to store the results of anomaly detection in actuators
    all_threshold_act = np.zeros(len(df_test), dtype=float)
    # List of arrays to store the threshold over time
    thresholds_sen = []
    # List of arrays to store the location of human interventions for debugging
    human_idx_sen = []


    for ix_feat_sen, feat_sen in enumerate(FEATURES_OUT_SEN):
        # Create threshold prediction model
        error_mdl = PredErrorModel(window_size_in=WINDOW_PAST, window_size_out=WINDOW_PRESENT)
        error_mdl.to(device)
        # load best parameters
        error_mdl.load_state_dict(pred_error_chk[ix_feat_sen])
        error_opt = torch.optim.SGD(error_mdl.parameters(), lr=THRESHOLD_LR)
        pred_error_model.append(error_mdl)
        pred_error_optimizer.append(error_opt)

        mdl = ModelSensors(n_devices_out=len(feat_sen))
        mdl.to(device)
        mdl.load_state_dict(model_checkpoint[ix_feat_sen])
        # Output sections are tuned on every human intervention
        for name, param in mdl.named_children():
            for p in param.parameters():
                p.requires_grad = True
        model_sensors.append(mdl)
        # Prepare the list of arrays
        all_predicted_sen.append(np.zeros(len(df_test), dtype=float))
        human_idx_sen.append(np.zeros(len(df_test), dtype=float))
        all_threshold_sen.append(np.zeros(len(df_test), dtype=float))
        thresholds_sen.append(np.zeros(len(df_test), dtype=float))
        
    # database_actuators refers to the database A (see the paper)
    # Fill the database A  with the normal combination of actuator states
    database_actuators = np.concatenate((df_train[FEATURES_OUT_ACT_FLAT].values, df_val[FEATURES_OUT_ACT_FLAT].values))
    database_actuators, indices, unique_counts = np.unique(database_actuators, axis=0, return_index=True, return_counts=True)

    # According to : https://www.researchgate.net/publication/305809559
    # Some of the attacks have a stronger effect on the dynamics of system and causing more time
    # for the system to stabilize (after the attack). Simpler attacks, such as those that effect flow rates,
    # require less time to stabilize. Also, some attacks do not take effect immediately (attack impact is seen after the attack's end).
    # Based on that, attack impact is considered as part of the attack, and we avoid human intervention on the period just after the attack
    for idx, (idx_strt, idx_end) in enumerate(attack_indices):
        if idx == 0:
            attack_impact_array = np.arange(idx_end, idx_end + (idx_end - idx_strt))
        else:
            attack_impact_array = np.concatenate((attack_impact_array, np.arange(idx_end, idx_end + (idx_end - idx_strt))))
    attack_impact_array = np.unique(attack_impact_array)

    all_labels_threshold = np.copy(all_labels).astype(int)
    # Attack impact is considered as part of the attack
    all_labels_threshold[attack_impact_array] = 1

    before = time.time()
    human_inter_counter = 0
    actuation_alarm = 0

    # Loop over the test dataset
    # Note that DAICS produces a decision for every batch
    for batch_idx in range(0, len(idx_test_wt) // BATCH_SIZE):
        print('\rBatch Index [%d] Human Intervention Counter [%d]'%(batch_idx, human_inter_counter), end="")

        # The first and last index in the current batch
        apply_thr_start = idx_test_wt[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE][0, 0]
        apply_thr_end = idx_test_wt[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE][-1, 0]
        # Prepare W_{out} of each output section
        window_t_sen = []
        for feat_sen in FEATURES_OUT_SEN:
            win_t_sen = df_test.iloc[idx_test_wt[batch_idx * BATCH_SIZE:  batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].loc[:, feat_sen].values.reshape(BATCH_SIZE, WINDOW_PRESENT, -1)
            win_t_sen = torch.from_numpy(win_t_sen).float().to(device)
            window_t_sen.append(win_t_sen)
        # Prepare W_{in}
        window_t_1 = df_test.iloc[idx_test_wt_1[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE].flatten()].values.reshape(BATCH_SIZE, WINDOW_PAST, -1)
        # Actuator combinations in the current batch
        window_t_actuator = df_test.iloc[np.unique(idx_test_wt[batch_idx * BATCH_SIZE:  batch_idx * BATCH_SIZE + BATCH_SIZE].flatten())].loc[:, FEATURES_OUT_ACT_FLAT].values[:BATCH_SIZE, :]
        window_t_1 = torch.from_numpy(window_t_1).float().to(device)
        # Calculate output at DL4
        y_t = model_f_extractor(window_t_1.float().to(device))
        # Anomaly detection in actuators
        dim1, dim2 = window_t_actuator.shape
        set_database_actuators = set(map(tuple, database_actuators))
        # List of actuators combinations that do not exist in database_actuators
        tmp = [(set(map(tuple, np.expand_dims(x_window_t_actuator, axis=0))) & set_database_actuators) == set() for  x_window_t_actuator in window_t_actuator]
        # If there are new combinations, report an anomaly alarm
        if np.any(tmp):
            actuation_alarm = 1
            # Set the indices of the novel combinations to one (i.e. anomalous)
            all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)] = tmp  
        else:
            actuation_alarm = 0
        # Prepare indices to predict the threshold
        idx_threshold = (np.arange(apply_thr_start, apply_thr_end + 1) - HORIZON)[:, None] - np.arange(1, WINDOW_PAST + 1)
        # Loop over the output sections of the wide and deep neural network
        for out_sec in range(len(FEATURES_OUT_SEN)):
            # Predict the normal sensor readings
            y_t_sen = model_sensors[out_sec](y_t)
            # Calculate prediction error
            pred_error_sen = test_loss_function(y_t_sen, window_t_sen[out_sec])
            # Store the prediction error for further processing
            all_predicted_sen[out_sec][idx_test_wt[batch_idx * BATCH_SIZE: batch_idx * BATCH_SIZE + BATCH_SIZE, 0]] = pred_error_sen.detach().cpu().numpy()[:, 0]
                            
            # Smooth the prediction error using a median filter
            all_predicted_sen[out_sec][apply_thr_start - W_ANOMALY * 2: apply_thr_end + 1] = scipy.signal.medfilt(all_predicted_sen[out_sec][apply_thr_start - W_ANOMALY * 2: apply_thr_end + 1], kernel_size=MED_FILTER_LAG)
            # Prepare W_{in} of the threshold prediction model
            threshold_wt_1 = torch.from_numpy(all_predicted_sen[out_sec][idx_threshold][:, :, None]).float().to(device)
            pred_error_model[out_sec].zero_grad()
            threshold = pred_error_model[out_sec](threshold_wt_1).abs()
            thresh_loss = threshold_loss_function(torch.squeeze(threshold, dim=2), pred_error_sen)
            # One threshold SGD step
            thresh_loss.backward(retain_graph=True)
            pred_error_optimizer[out_sec].step()
            # Use the maximum predicted threshold for anomaly detection
            threshold = torch.max(threshold, dim=1)[0].detach().cpu().numpy()
            # If the threshold is zero, repeat the process
            if np.all(threshold == 0):
                pred_error_model[out_sec].zero_grad()
                pred_error_model[out_sec].apply(init_weights)
                threshold = pred_error_model[out_sec](threshold_wt_1).abs()
                thresh_loss = threshold_loss_function(torch.squeeze(threshold, dim=2), pred_error_sen)
                thresh_loss.backward(retain_graph=True)
                pred_error_optimizer[out_sec].step()
                threshold = torch.max(threshold, dim=1)[0].detach().cpu().numpy()
            # Sum the predicted threshold to the threshold base
            threshold = threshold + thresh_sen_base[out_sec]
            # Store the calculated threshold for debugging
            thresholds_sen[out_sec][apply_thr_start:apply_thr_end + 1] = np.squeeze(threshold)
            # Apply W_ANOMALY on prediction error of sensors
            # (- np.arange(W_ANOMALY)) means to use the previous W_ANOMALY samples
            idx_win_thr = np.arange(apply_thr_start, apply_thr_end + 1)[:, None] - np.arange(W_ANOMALY)
            # If error exceeds the threshold for W_ANOMALY samples (seconds), consider the current reading as anomaly
            # Store the anomaly detection in sensors of the current output section
            all_threshold_sen[out_sec][np.arange(apply_thr_start, apply_thr_end + 1)] = np.all(all_predicted_sen[out_sec][idx_win_thr] > threshold, 1)
        # Note that alarms with period <= W_GRACE will be silenced
        if np.any(all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)] == 1) and np.count_nonzero(all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)]) <= W_GRACE:
            # add unknown actuator combinations to database_actuators
            database_actuators = np.concatenate((database_actuators, window_t_actuator), axis=0)
            database_actuators = np.unique(database_actuators, axis=0)
            # The alarm is silenced
            all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)] = 0
        # This step Simulates human intervention due to actuation alarms
        # If there are new actuator combinations and the records are not marked as attack, add the new combinations to the database
        if np.any(all_threshold_act[np.arange(apply_thr_start, apply_thr_end + 1)] == 1) and np.all(all_labels_threshold[np.arange(apply_thr_start, apply_thr_end + 1)] == 0):
            # add unknown actuator combinations to database_actuators
            database_actuators = np.concatenate((database_actuators, window_t_actuator), axis=0)
            database_actuators = np.unique(database_actuators, axis=0) 
            # Increment human intervention counter
            human_inter_counter += 1 
        
        for out_sec in  range(len(FEATURES_OUT_SEN)):
        ###=====================================Tune Sections===========================================================
            # Note that alarms with period <= W_GRACE will be silenced and the model get fine-tuned
            if np.any(all_threshold_sen[out_sec][np.arange(apply_thr_start, apply_thr_end + 1)] == 1) and np.count_nonzero(all_threshold_sen[out_sec][np.arange(apply_thr_start, apply_thr_end + 1)]) <= W_GRACE:
                # Prepare the optimizer
                ftune_optimizer = torch.optim.SGD(model_sensors[out_sec].parameters(), lr=OUTPUT_LR, momentum=0.9, dampening=0.9, weight_decay=0.001)
                ftune_scheduler = ReduceLROnPlateau(ftune_optimizer, verbose=False)
                # Fine-tune the output section
                for epoch in range(T_EPOCHS):
                    model_sensors[out_sec].zero_grad()
                    f_extracted = model_f_extractor(window_t_1[:-4].float().to(device))
                    y_t_sen = model_sensors[out_sec](f_extracted)
                    loss = ftune_loss_function(y_t_sen, window_t_sen[out_sec][:-4])
                    loss.backward()
                    ftune_optimizer.step()
                    ftune_scheduler.step(loss)
                # The alarm is silenced
                all_threshold_sen[out_sec][np.arange(apply_thr_start, apply_thr_end + 1)] = 0

            # This line simulates the human intervention process due to alarms in sensors
            if np.any(all_threshold_sen[out_sec][np.arange(apply_thr_start, apply_thr_end + 1)] == 1) and np.all(all_labels_threshold[np.arange(apply_thr_start, apply_thr_end + 1)] == 0):
                # Increment the human intervention counter
                human_inter_counter += 1
                # Flag the indices of human intervention for debugging purposes
                human_idx_sen[out_sec][np.arange(apply_thr_start, apply_thr_end + 1)] = 1
                ftune_optimizer = torch.optim.SGD(model_sensors[out_sec].parameters(), lr=OUTPUT_LR, momentum=0.9, dampening=0.9, weight_decay=0.001)
                ftune_scheduler = ReduceLROnPlateau(ftune_optimizer, verbose=False)
                # Fine-tune the output section
                for epoch in range(T_EPOCHS):
                    model_sensors[out_sec].zero_grad()
                    w_t_1 = window_t_1.detach().clone()
                    f_extracted = model_f_extractor(w_t_1[:-4].float().to(device))
                    y_t_sen = model_sensors[out_sec](f_extracted)
                    loss = ftune_loss_function(y_t_sen, window_t_sen[out_sec][:-4])
                    loss.backward()
                    ftune_optimizer.step()
                    ftune_scheduler.step(loss)


    after = time.time()
    print("\nTime elapsed on the attack dataset:", after - before)
    print("Number of human interventions: ", human_inter_counter)
    # end for enumerate(dl_test):                                            
    # Combining the anomaly detection results of all output sections
    all_threshold = np.zeros(len(df_test), dtype=float)
    for all_thr in all_threshold_sen:
        all_threshold = np.logical_or(all_threshold, all_thr)
    # Combine with anomaly detection in actuators
    all_threshold = np.logical_or(all_threshold, all_threshold_act)
    # Make sure that all arrays have the same length
    all_threshold = all_threshold[: len(all_labels)]
    all_labels_threshold = np.copy(all_labels)[: len(all_labels)]
    # Attack impact is part of the attack
    # Ref: http://dx.doi.org/10.1145/3196494.3196546
    all_labels_threshold[attack_impact_array] = all_threshold[attack_impact_array]
    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels_threshold, all_threshold).ravel()
    print("accuracy {}".format(accuracy_score(all_labels_threshold, all_threshold)))
    print("precision {}".format(precision_score(all_labels_threshold, all_threshold)))
    print("recall {}".format(recall_score(all_labels_threshold, all_threshold)))
    print("false positive rate", fp/(fp+tn))
    print("false negative rate", fn/(tp+fn))
    print("\x1b[6;30;42m f1_oneclass_score \x1b[0m {}".format(f1_score(all_labels_threshold, all_threshold)))
    # Count detected attacks
    detected_counter = 0
    for idx, (idx_strt, idx_end) in enumerate(attack_indices):
        print("f1 for attack #", str(idx + 1) + " = " +
                str(f1_score(
                    all_labels_threshold[range(idx_strt, idx_end + (idx_end - idx_strt))],
                    all_threshold[range(idx_strt, idx_end + (idx_end - idx_strt))])))
        print("recall for attack #", str(idx + 1) + " = " +
                str(recall_score(
                    all_labels_threshold[range(idx_strt, idx_end + (idx_end - idx_strt))],
                    all_threshold[range(idx_strt, idx_end + (idx_end - idx_strt))])))
        if recall_score(all_labels_threshold[range(idx_strt, idx_end + (idx_end - idx_strt))],
                        all_threshold[range(idx_strt, idx_end + (idx_end - idx_strt))]) > 0.001:
            detected_counter += 1
    print("Detected attacks=", detected_counter)
    # Plot prediction error vs threshold vs true labels for debugging
    for i in range(len(FEATURES_OUT_SEN)):
        lbls = all_labels * np.amax(all_predicted_sen[i]) + np.amin(all_predicted_sen[i])
        tune_plt = human_idx_sen[i] * np.amax(all_predicted_sen[i]) + np.amin(all_predicted_sen[i])
        plt.figure(figsize=(300,3))
        plt.plot(all_predicted_sen[i], label="all_predicted_sen[" + str(i) + "]", c='b')
        plt.plot(lbls, label="Labels", c='r')
        plt.plot(tune_plt, label="Tune", c='g')
        plt.plot(thresholds_sen[i], label="Threshold", linestyle="--", c='xkcd:orange')
        plt.legend(loc='lower left', fancybox=True)
        plt.xlabel("Time (s)")
        plt.ylabel("MSE")
        plt.grid(linestyle='--')
        plt.show()
    # Plot detected anomalies in actuators vs true labels for debugging
    plt.figure(figsize=(300,3))
    plt.plot(all_labels * 1.2, label="Labels", c='r')
    plt.plot(all_threshold_act, label="all_threshold_act", c='xkcd:orange')
    plt.legend(loc='lower left', fancybox=True)
    plt.xlabel("Time (s)")
    plt.grid(linestyle='--')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Execute the DAICS framework!")
    parser.add_argument("--dataset", type=str, help="Dataset name (swat, wadi)")
    parser.add_argument("--mode", type=str, help="Execution mode (train, eval)")



    args = parser.parse_args()
    


    seed_torch()
    if args.dataset in ['swat']:
        swat_execute(mode=args.mode)
    if args.dataset in ['wadi']:
        wadi_execute(mode=args.mode)

if __name__ == "__main__":
    main()