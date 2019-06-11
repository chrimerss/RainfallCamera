import torch
import torch.nn as nn
import torch.utils.data as udata
import h5py
import os
import cv2
import numpy as np
import random


def make_bg_data():
	data_path= 'datasets/train'
	in_dir= ['heavy','normal','no_rain']
	input_data= h5py.File('input_BG_Dis.h5','w')
	target_data= h5py.File('target_BG_Dis.h5','w')
	key=0

	for path in in_dir:
		if path=='heavy' or path=='normal':
			label=[1]
		else:
			label=[0]
		data_dir= os.path.join(data_path,path)
		imgs= os.listdir(data_dir)
		for img in imgs:

			img_path= os.path.join(data_dir, img)
			src= cv2.imread(img_path)
			src= cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
			if np.array(src.shape).any()<401:
				src= cv2.resize(src, (401,401))
				src= src[:,:,0]
			#crop
			else:
				src= src[600:1001,300:701,0]
			src= src[np.newaxis,:,:]
			assert src.shape==(1,401,401), 'get %s'%(str(src.shape))
			input_data.create_dataset(str(key),data=np.array(src)/255.)
			target_data.create_dataset(str(key),data=np.array(label))
			key+=1

	input_data.close()
	target_data.close()

	return None

def make_rain_data():
	base_dir='datasets'
	in_dir=['rainstreak','noise']
	inputs= h5py.File('input_rain_Dis.h5','w')
	targets= h5py.File('target_rain_Dis.h5','w')
	key=0
	for path in in_dir:
		data_path= os.path.join(base_dir, path)
		if path=='rainstreak':
			label=[1]
		else:
			label=[0]

		img_list= os.listdir(data_path)
		for img in img_list:
			src= cv2.imread(os.path.join(data_path, img))
			src= cv2.resize(src, (401,401))[np.newaxis,:,:,0]

			# cv2.imshow('rainstreak',src[:,:,0])
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			assert src.shape==(1,401,401), 'data shape%s'%str(src.shape)
			inputs.create_dataset(str(key), data=np.array(src)/255.)
			targets.create_dataset(str(key), data= np.array(label))

			key+=1
	inputs.close()
	targets.close()
	print('total training samples: ',key)

	return None
			

class DataSet(udata.Dataset):
	def __init__(self, datatype='bg'):
		#open h5 file
		super(DataSet, self).__init__()
		self.datatype= datatype
		if datatype=='bg':
			inputs= h5py.File('input_BG_Dis.h5','r')
			targets= h5py.File('target_BG_Dis.h5','r')
		elif datatype== 'rain':
			inputs= h5py.File('input_rain_Dis.h5','r')
			targets= h5py.File('target_rain_Dis.h5','r')

		self.keys= list(inputs.keys())
		random.shuffle(self.keys)

		inputs.close()
		targets.close()

	def __getitem__(self, index):
		if self.datatype=='bg':
			inputs= h5py.File('input_BG_Dis.h5','r')
			targets= h5py.File('target_BG_Dis.h5','r')
		elif self.datatype=='rain':
			inputs= h5py.File('input_rain_Dis.h5','r')
			targets= h5py.File('target_rain_Dis.h5','r')

		key= self.keys[index]
		train= np.array(inputs[key])
		target= np.array(targets[key]).astype(np.int64)
		inputs.close()
		targets.close()
		# print(train.shape, target.shape)

		return torch.Tensor(train), torch.Tensor(target)

	def __repr__(self):
		print('total samples: ', len(self.keys))

	def __len__(self):
		return len(self.keys)

class DataSet_GAN(udata.Dataset):
	def __init__(self, data_path='datasets', tsize=2):
		super(DataSet_GAN, self).__init__()

		self.data_path= data_path

		train_data= os.path.join(self.data_path,'train_2.h5')
		train_data= h5py.File(train_data,'r')

		self.keys= list(train_data.keys())
		random.shuffle(self.keys)

		train_data.close()

	def __len__(self):

		return len(self.keys)

	def __getitem__(self, index):
		
		train_data= os.path.join(self.data_path,'train_2.h5')
		train_data= h5py.File(train_data,'r')
		key= self.keys[index]
		assert train_data[key].shape==(2,401,401),\
				'expected (2,401,401) but got %s'%str(train_data[key].shape)
		prev_data= train_data[key][0,:,:]
		now_data= train_data[key][1,:,:]
		train_data.close()


		return torch.Tensor(prev_data[np.newaxis,:,:]), torch.Tensor(now_data[np.newaxis,:,:])


if __name__=="__main__":
	make_bg_data()

