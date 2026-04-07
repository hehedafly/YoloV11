# coding:utf-8
 
import os
import shutil
import random
import argparse
import numpy as np

from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
 

# MainDatasetPaths = [r'D:\Unity\PythonFiles\YoloV8\OutputMouseBodyPic1012124927', r'D:\Unity\PythonFiles\YoloV8\OutputMouseBodyPic1012140021', r'D:\Unity\PythonFiles\YoloV8\OutputStaticObjectPic1012124927']
# samplePerDataset = [1, 1, 0.2]
MainDatasetPaths = [r'E:\pythonFiles\YoloV8\PoseExtract0113133129']
samplePerDataset = [1]
# parser = argparse.ArgumentParser()
# parser.add_argument('--rawDataPath', default='D:/Unity/PythonFiles/YoloV8/PoseExtract0407120327', type=str, help='All File path')
# parser.add_argument('--txt_path', default='D:/Unity/PythonFiles/YoloV8/YoloPoseTrainData/labels', type=str, help='output txt label path')
# parser.add_argument('--img_path', default='D:/Unity/PythonFiles/YoloV8/YoloPoseTrainData/images', type=str, help='output img path')
# opt = parser.parse_args()
# RawDataPath = opt.rawDataPath
# txtsavepath = opt.txt_path
# imgsavepath = opt.img_path
# RawDataPath = ''
txtsavepath = 'E:/pythonFiles/YoloV8/YoloTrainData/labels'
imgsavepath = 'E:/pythonFiles/YoloV8/YoloTrainData/images'
 
trainval_percent = 0.9
train_percent = 0.9

dirs = [txtsavepath+"/train", txtsavepath+"/val", txtsavepath+"/test", imgsavepath+"/train", imgsavepath+"/val", imgsavepath+"/test"]
for path in dirs:
    if not os.path.exists(path):
        os.makedirs(path)

train = []
trainval = []
lastNum = 0
totalFile = []
totalFilePaths = []
isPose = True
for i, RawDataPath in enumerate(MainDatasetPaths):
    tempFile = []
    with os.scandir(RawDataPath) as entries:
            for entry in entries:
                if entry.name.endswith("ROI.png" if not isPose else "pose.png") and random.randrange(0, 10, 1) < (samplePerDataset[i] * 10):
                # if not entry.name.endswith(".png"):
                    totalFilePaths.append(entry.path)
                    totalFile.append(entry.name[:-7 if not isPose else -8])
                    tempFile.append(entry.name[:-7 if not isPose else -8])

    num = len(tempFile)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval += list(np.array(random.sample(range(num), tv)) + lastNum)
    train += list(np.array(random.sample(trainval, tr)) + lastNum)
    lastNum += num

for i, name in enumerate(totalFile):
    if i in trainval:
        # file_trainval.write(name)
        fileBase:str = totalFilePaths[i]
        fileBase = ".".join(fileBase.split("ROI.")[:-1]) if not isPose else ".".join(fileBase.split("pose.")[:-1])
        if i in train:
            shutil.copy(fileBase + ".jpg", dirs[3]+"/"+name+".jpg")
            shutil.copy(fileBase + ".txt", dirs[0]+"/"+name+".txt")
            # file_train.write(name)
        else:
            shutil.copy(fileBase + ".jpg", dirs[4]+"/"+name+".jpg")
            shutil.copy(fileBase + ".txt", dirs[1]+"/"+name+".txt")
            # file_val.write(name)
    else:
        shutil.copy(fileBase + ".jpg", dirs[5]+"/"+name+".jpg")
        shutil.copy(fileBase + ".txt", dirs[2]+"/"+name+".txt")
        # file_test.write(name)