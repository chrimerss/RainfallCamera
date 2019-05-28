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


def data_prep():
	video_path= '../videos/20180401/20180401_152716_8BB3.mkv'
	cap= cv2.VideoCapture(video_path)
	ind=0
	dst= 'datasets/20180401'
	localtime= video_path.split(os.sep)[-1].split('.')[0]
	date, time, _ = localtime.split('_')
	localtime= datetime.datetime.strptime(date+time, '%Y%m%d%H%M%S')
	first=True
	while True:
		ind+=1
		ret, frame= cap.read()
		if not ret or ind>250:
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
	folders= os.listdir(src)
	key=1
	train_path= os.path.join('.','train.h5')
	train_data= h5py.File(train_path,'w')

	for folder in folders:
		img_names= os.listdir(os.path.join(src, folder))
		ind=0 
		while True:
			_data= []
			try:
				for it in range(tsize):
					img= cv2.imread(os.path.join(src,folder,img_names[ind+it]))
					img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.
					_data.append(img)
				ind+=1

			except IndexError:
				break

			assert np.array(_data).shape==(2,401,401), \
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

if __name__=='__main__':
	pyh5()