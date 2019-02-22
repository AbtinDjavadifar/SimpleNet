#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:18:55 2019

@author: aeroclub
"""

import shutil, random, os

Wrinkle_dirpath_source="/home/aeroclub/Abtin/CASE/Simple_Net/DLR_Dataset/Wrinkle/"
Fabric_dirpath_source="/home/aeroclub/Abtin/CASE/Simple_Net/DLR_Dataset/Fabric/"
Gripper_dirpath_source="/home/aeroclub/Abtin/CASE/Simple_Net/DLR_Dataset/Gripper/"
Background_dirpath_source="/home/aeroclub/Abtin/CASE/Simple_Net/DLR_Dataset/Background/"

Wrinkle_dirpath_dest_train="/home/aeroclub/Abtin/CASE/Simple_Net/Training_Images/Training_Set/Wrinkle/"
Fabric_dirpath_dest_train="/home/aeroclub/Abtin/CASE/Simple_Net/Training_Images/Training_Set/Fabric/"
Gripper_dirpath_dest_train="/home/aeroclub/Abtin/CASE/Simple_Net/Training_Images/Training_Set/Gripper/"
Background_dirpath_dest_train="/home/aeroclub/Abtin/CASE/Simple_Net/Training_Images/Training_Set/Background/"

Wrinkle_dirpath_dest_test="/home/aeroclub/Abtin/CASE/Simple_Net/Training_Images/Test_Set/Wrinkle/"
Fabric_dirpath_dest_test="/home/aeroclub/Abtin/CASE/Simple_Net/Training_Images/Test_Set/Fabric/"
Gripper_dirpath_dest_test="/home/aeroclub/Abtin/CASE/Simple_Net/Training_Images/Test_Set/Gripper/"
Background_dirpath_dest_test="/home/aeroclub/Abtin/CASE/Simple_Net/Training_Images/Test_Set/Background/"

k=4
num_wrinkle=250*k
num_fabric=150*k
num_gripper=150*k
num_background=200*k


filenames = random.sample(os.listdir(Wrinkle_dirpath_source), num_wrinkle+50)

for fname in filenames[0:num_wrinkle]:
    srcpath = os.path.join(Wrinkle_dirpath_source, fname)
    shutil.copy(srcpath, Wrinkle_dirpath_dest_train)

for fname in filenames[num_wrinkle:]:
    srcpath = os.path.join(Wrinkle_dirpath_source, fname)
    shutil.copy(srcpath, Wrinkle_dirpath_dest_test)    
 
    
filenames = random.sample(os.listdir(Fabric_dirpath_source), num_fabric+50)

for fname in filenames[0:num_fabric]:
    srcpath = os.path.join(Fabric_dirpath_source, fname)
    shutil.copy(srcpath, Fabric_dirpath_dest_train)

for fname in filenames[num_fabric:]:
    srcpath = os.path.join(Fabric_dirpath_source, fname)
    shutil.copy(srcpath, Fabric_dirpath_dest_test)        
    
    
filenames = random.sample(os.listdir(Gripper_dirpath_source), num_gripper+50)

for fname in filenames[0:num_gripper]:
    srcpath = os.path.join(Gripper_dirpath_source, fname)
    shutil.copy(srcpath, Gripper_dirpath_dest_train)

for fname in filenames[num_gripper:]:
    srcpath = os.path.join(Gripper_dirpath_source, fname)
    shutil.copy(srcpath, Gripper_dirpath_dest_test)        
    
filenames = random.sample(os.listdir(Background_dirpath_source), num_background+50)

for fname in filenames[0:num_background]:
    srcpath = os.path.join(Background_dirpath_source, fname)
    shutil.copy(srcpath, Background_dirpath_dest_train)

for fname in filenames[num_background:]:
    srcpath = os.path.join(Background_dirpath_source, fname)
    shutil.copy(srcpath, Background_dirpath_dest_test)      
    