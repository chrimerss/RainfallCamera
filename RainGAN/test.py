import torch
import torch.nn as nn
import torch.utils.data as udata
from dataprep import Data_train_Gen
import cv2
import numpy as np

# data= GanData()
# print(data[2])
data= Data_train_Gen()
inputs, targets= data[3]
input= (inputs[0,:,:,:].squeeze().numpy()*255.).astype(np.uint8)
target= (targets[0,:,:,:].squeeze().numpy()*255.).astype(np.uint8)
cv2.imshow('input', input)
cv2.imshow('ground_truth', target)
cv2.waitKey(0)
cv2.destroyAllWindows()