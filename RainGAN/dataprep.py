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

def make_syn_paired():
	"""make synthetic paired rain for training G"""
	base_dir= 'datasets'
	dir_list= ['rain_data_train_Heavy', 'rain_data_train_Light']
	inputs= h5py.File('datasets/synthetic_data_paired_input.h5','w')
	targets= h5py.File('datasets/synthetic_data_paired_target.h5', 'w')

	for each in dir_list:
		curr_path= os.path.join(base_dir, each)
		for repo in os.listdir(curr_path):
			key= 0
			print('Processing repo: ',repo,'\n--------------------------')
			imgs= sorted(os.listdir(os.path.join(curr_path, repo)))
			for img in imgs:
				print('processing :',img)
				src= cv2.imread(os.path.join(curr_path, repo, img))
				if src is None:
					raise ValueError('image not read correctly!')
				src= cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)[:,:,0]/255.
				if (np.array(src.shape)<401).any():
					src= cv2.resize(src, (401,401))
				else:
					m,n= src.shape
					rand_i= np.random.randint(0,m-401)
					rand_j= np.random.randint(0,n-401)
					src= src[m:m+401,n:n+401]

				if repo =='rain':
					if each=='rain_data_train_Heavy':
						keyword= 'heavy-%s'%img.split('.')[0][:-2]
					elif each=='rain_data_train_Light':
						keyword= 'light-%s'%img.split('.')[0][:-2]
					else:
						raise NotImplementedError('name doesnot exist!')
					inputs.create_dataset(keyword, data=np.array(src)[np.newaxis,:,:])

				elif repo=='norain':
					if each=='rain_data_train_Heavy':
						keyword= 'heavy-%s'%img.split('.')[0]
					elif each=='rain_data_train_Light':
						keyword= 'light-%s'%img.split('.')[0]
					else:
						raise NotImplementedError('name doesnot exist!')
					# print(keyword)
					targets.create_dataset(keyword, data=np.array(src)[np.newaxis,:,:])
				
				key+=1

	inputs.close()
	targets.close()

class Data_train_Gen(udata.Dataset):
	def __init__(self, datapath='datasets', tsize=2):
		self.datapath= datapath
		inputs= h5py.File(os.path.join(self.datapath, 'synthetic_data_paired_input.h5'), 'r')
		
		self.keys= list(inputs.keys())

		self.tsize=tsize
		
		random.shuffle(self.keys)
		inputs.close()

	def __len__(self):
		return len(self.keys)-1

	def __getitem__(self, index):
		# if index==0:
		key_prev= self.keys[index]
		key_now= self.keys[index+1]


		inputs= h5py.File(os.path.join(self.datapath, 'synthetic_data_paired_input.h5'), 'r')
		targets= h5py.File(os.path.join(self.datapath, 'synthetic_data_paired_target.h5'), 'r')
		input_prev= np.array(inputs[key_prev])
		target_prev= np.array(targets[key_prev])
		input_now= np.array(inputs[key_now])
		target_now= np.array(inputs[key_now])
		inputs.close()
		targets.close()
		input= np.concatenate([input_prev, input_now], axis=0)[:, np.newaxis,:,:]
		target= np.concatenate([target_prev, target_now], axis=0)[:, np.newaxis,:,:]

		assert input.shape==(2,1,401,401),'expected (2,1,401,401), but get %s instead'%(str(input.shape))
		assert target.shape==(2,1,401,401),'expected (2,1,401,401), but get %s instead'%(str(target.shape))

		return torch.Tensor(input), torch.Tensor(target)

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

class GenData(udata.Dataset):
	def __init__(self, data_path='datasets', tsize=2):
		super(GenData, self).__init__()

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

class DisData_rain(udata.Dataset):
	def __init__(self, datapath='datasets'):
		super(DisData_rain, self).__init__()
		self.datapath= datapath
		inputs= h5py.File(os.path.join(self.datapath, 'dis-train-rain-input.h5'), 'r')
		# targets= h5py.File(os.path.join(self.datapath, 'dis-train-rain-label.h5'),'r')

		self.keys= list(inputs.keys())

		random.shuffle(self.keys)

		inputs.close()
		# targets.close()
	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):

		inputs= h5py.File(os.path.join(self.datapath, 'dis-train-rain-input.h5'), 'r')
		targets= h5py.File(os.path.join(self.datapath, 'dis-train-rain-label.h5'),'r')

		input= np.array(inputs[self.keys[index]])
		target= np.array(targets[self.keys[index]])

		assert input.shape==(1,401,401),'expected (1,401,401) but found %s'%(str(input.shape))
		inputs.close()
		targets.close()

		return torch.Tensor(input), torch.Tensor(target)

class DisData_bg(udata.Dataset):
	def __init__(self, datapath='datasets'):

		super(DisData_bg, self).__init__()
		self.datapath= datapath
		inputs= h5py.File(os.path.join(self.datapath, 'dis-train-bg-input.h5'), 'r')
		# targets= h5py.File(os.path.join(self.datapath, 'dis-train-rain-label.h5'),'r')

		self.keys= list(inputs.keys())

		random.shuffle(self.keys)

		inputs.close()
		# targets.close()
	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):

		inputs= h5py.File(os.path.join(self.datapath, 'dis-train-bg-input.h5'), 'r')
		targets= h5py.File(os.path.join(self.datapath, 'dis-train-bg-label.h5'),'r')

		input= np.array(inputs[self.keys[index]])
		target= np.array(targets[self.keys[index]])

		assert input.shape==(1,401,401),'expected (1,401,401) but found %s'%(str(input.shape))
		inputs.close()
		targets.close()

		return torch.Tensor(input), torch.Tensor(target)

class GanData(udata.Dataset):

	def __init__(self):
		super(GanData, self).__init__()

		self.data_g= GenData()
		self.data_d_rain= DisData_rain()
		self.data_d_bg= DisData_bg()


	def __len__(self):
		return len(self.data_g.keys)

	def __getitem__(self, index):

		g_data= self.data_g[index]
		rain_data= self.data_d_rain[index]
		bg_data= self.data_d_bg[index]

		return g_data, rain_data, bg_data


if __name__=="__main__":
	make_syn_paired()

