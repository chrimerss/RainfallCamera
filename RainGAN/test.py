import torch
import torch.nn as nn
import torch.utils.data as udata
from dataprep import GanData, DataSet_G, DataSet_D_rain, DataSet_D_bg

# data= GanData()
# print(data[2])
data= DataSet_D_bg()
print(data[3])