import cv2
import numpy as np
import datetime
import os
from numba import jit
import torch.utils.data as udata
import h5py
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def data_prep():
	video_path= '../videos/20180401/20180401_153226_6AD4.mkv'
	cap= cv2.VideoCapture(video_path)
	ind=0
	dst= 'datasets/20180401'
	localtime= video_path.split(os.sep)[-1].split('.')[0]
	# print(localtime)
	date, time, _ = localtime.split('_')
	localtime= datetime.datetime.strptime(date+time, '%Y%m%d%H%M%S')
	first=True
	while True:
		ind+=1
		ret, frame= cap.read()
		if not ret or ind>2:
			break
		if first:
			rows, cols= autocrop(frame, window_size=(401,401))
			first=False
		frame= frame[rows[0]:rows[1],cols[0]:cols[1]]
		if ind>50:
			cv2.imwrite(os.path.join(dst,'%s.png'%localtime.strftime('%Y%m%d%H%M%S')),frame)
		localtime+= datetime.timedelta(seconds=1)


def autocrop(src, window_size=(300,300)):
	#with numba, the cost reduces from 92 seconds to 29 seconds.
	@jit(nopython=True)
	def moving_window(h,w, min_val, window_size):
    # val= overdetection(src, 2)
    # src[src<=val]=0
    # src[src>val]=255
		for i in range(h-window_size[0]):
			for j in range(w-window_size[1]):
				tot= src[i:window_size[0]+i, j:j+window_size[1]].sum()
				if tot<min_val:
					min_val= tot
					rows= (i, window_size[0]+i)
					cols= (j,j+window_size[1])

		return rows, cols
	src= cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	h,w= src.shape
	min_val= np.inf

	rows, cols= moving_window(h,w,min_val,window_size)

	return rows, cols

def pyh5(tsize=2):
	src= 'datasets/'
	folders= ['20180401','20181208','20181211','20181212','20181221']
	key=0
	train_path= os.path.join('.','train_2.h5')
	train_data= h5py.File(train_path,'w')

	for folder in folders:
		img_names= os.listdir(os.path.join(src, folder))
		ind=0 
		while True:
			_data= []
			try:
				for it in range(tsize):
					img= cv2.imread(os.path.join(src,folder,img_names[ind+it]))
					img= cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)/255.
					_data.append(img[:,:,0])
				ind+=1

			except IndexError:
				break

			assert np.array(_data).shape==(tsize,401,401), \
					'expected (2,401,401,3), but got %s'%str(np.array(_data).shape)
			train_data.create_dataset(str(key),data=np.array(_data))
			key+=1

	print('total training samples: ',key)
	train_data.close()

class Dataset(udata.Dataset):
	def __init__(self, data_path='datasets/', tsize=2):
		super(Dataset, self).__init__()

		self.data_path= data_path

		train_data= os.path.join('train.h5')
		train_data= h5py.File(train_data,'r')

		self.keys= list(train_data.keys())
		random.shuffle(self.keys)

		train_data.close()

	def __len__(self):

		return len(self.keys)

	def __getitem__(self, index):
		
		train_data= os.path.join('train.h5')
		train_data= h5py.File(train_data,'r')
		key= self.keys[index]
		assert train_data[key].shape==(2,401,401),\
				'expected (2,401,401) but got %s'%str(train_data[key].shape)
		prev_data= train_data[key][0,:,:]
		now_data= train_data[key][1,:,:]
		train_data.close()


		return torch.Tensor(prev_data[np.newaxis,:,:]), torch.Tensor(now_data[np.newaxis,:,:])



def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

def init_weights(m):
	classname = m.__class__.__name__
	# for every convolution layer in a model..
	if classname.find('Conv') != -1:
		# apply a uniform distribution to the weights and a bias=0
		m.weight.data.uniform_(0.0, 1.0)
		m.bias.data.fill_(0)

class RainLoss(nn.Module):
	'''
	differientiable?
	'''
	def __init__(self, labmda1=0.1, labmda2=0.1, labmda3=0.4, labmda4=0.4, use_gpu=True):
		super(RainLoss,self).__init__()
		self.labmda1= labmda1
		self.labmda2= labmda2
		self.labmda3= labmda3
		self.labmda4= labmda4
		self.use_gpu= use_gpu
		if self.use_gpu:
			self.kernel_v= torch.Tensor([[0,-1,0],[0,0,0],[0,1,0]]).view(1,1,3,3).cuda()
			self.kernel_h= torch.Tensor([[0,0,0],[-1,0,1],[0,0,0]]).view(1,1,3,3).cuda()
		else:
			self.kernel_v= torch.Tensor([[0,-1,0],[0,0,0],[0,1,0]]).view(1,1,3,3)
			self.kernel_h= torch.Tensor([[0,0,0],[-1,0,1],[0,0,0]]).view(1,1,3,3)

	def forward(self, rain_prev, bg_prev, rain_now, bg_now ):

		zeros= torch.zeros(rain_now.size(), requires_grad=False).cuda() if self.use_gpu else torch.Tensor(rain_now.size(), requires_grad=False)
		sparsity= F.l1_loss(rain_now,zeros,reduction='sum')
		v_smooth= F.l1_loss(F.conv2d(rain_now, self.kernel_v,stride=1,padding=1),zeros,reduction='sum')
		h_smooth= F.l1_loss(F.conv2d(bg_now, self.kernel_h, stride=1,padding=1),zeros,reduction='sum')
		t_smooth= F.l1_loss(bg_now,bg_prev, reduction='sum')
		loss= self.labmda1*sparsity+ self.labmda2*v_smooth+self.labmda3*h_smooth+self.labmda4*t_smooth

		return loss

if __name__=='__main__':
	pyh5()
	# data_prep()