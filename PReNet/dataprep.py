import cv2
import numpy as np
import matplotlib.pyplot as plt
import dateutil
from glob import glob
from utils import addrain, autocrop_night, autocrop_day, normalize
import scipy.io
import os
import h5py
import torch.utils.data as udata
import random
import torch

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1

    return Y.reshape([endc, win, win, TotalPatNum])

def read_video(video_name):
	# where videos are stored:
   # change if video stores somewhere else
	video= cv2.VideoCapture(video_name)
	return video

def video2image(num_frames, video_name, store_img=True, crop=False):
	video= read_video(video_name)
	#get video infomation
	width= video.get(3)
	height= video.get(4)
	Frames= []
	ind=0
	first= True
	if width==1920 and height==1080:
		label='Full HD'
	else:
		label=f'resolution of video: {width, height}'
	while True:
		ret, frame= video.read()
		if ind==num_frames or not ret:
			break
		if first:
			window_size=(200,200) if height<300 or width<300 else (300,300)
			rows, cols= autocrop_day(frame, window_size)
			first=False
		Frames.append(frame[rows[0]:rows[1],cols[0]:cols[1],:])
		ind+=1

	Frames= np.array(Frames)

	return Frames.transpose(1,2,3,0)

def rand_video2image(num_frames, video_name, store_img=False):
	video= read_video(video_name)
	Frames= []
	ind=-1
	n_rand= np.random.choice(range(310),100, replace=False)
	while True:
		ind+=1
		ret, frame= video.read()
		if len(Frames)==num_frames or not ret:
			break
		elif ind in n_rand:
			Frames.append(frame)

	Frames= np.array(Frames)

	return Frames

def pick_video(path):
	videos= glob(os.path.join(path, '*.mkv'))
	video= np.random.choice(videos)

	return video

def syn_test(store_syn_img=False):
	window_size= (300,300) # you may want to modify this depending on the quality
	video= pick_video('D:\\CCTV\\rainfallcamera\\videos\\no-rain')
	Frames= rand_video2image(100, video,store_img=False)
	assert len(Frames.shape)==4,f"The shape of Frames is not consistent, expected 4 but {len(Frames.shape)} received"
	new_shape= (Frames.shape[0],300,300,3)
	# print(new_shape)
	# print('auto cropping image')
	# croped_rows, croped_cols= autocrop_day(cv2.cvtColor(Frames[0,:,:,:], cv2.COLOR_BGR2GRAY), window_size)
	# print(f'croped image: {croped_rows, croped_cols}')

	new_frames= np.zeros(new_shape, dtype=np.uint8)
	new_frames= Frames[:, 600:1000, 300:600,:].copy()
	for i in range(Frames.shape[0]):
		frame=  new_frames[i,:,:,:].copy()
		synthetic_img, _= addrain(frame)
		if store_syn_img:
			cv2.imwrite(f'D:\\CCTV\\rainfallcamera\\datasets\\rain\\rain-{i}.png', synthetic_img)
			cv2.imwrite(f'D:\\CCTV\\rainfallcamera\\datasets\\no-rain\\norain-{i}.png', new_frames[i,:,:,:])
	print('Synthetic rainfall added ...')

	return new_frames, synthetic_img
	#save as the .mat format
	# scipy.io.savemat('D:\\Radar Projects\\lizhi\\CCTV\\Videos\\20190110191017-rain.mat', mdict={'Rain': rain_streaks})
	# scipy.io.savemat('D:\\Radar Projects\\lizhi\\CCTV\\Videos\\20190110191017-img.mat', mdict={'Rain': new_frames})

def pyh5(win):
	rain_img= np.zeros((300,400,300,3), np.uint8)
	norain_img= np.zeros((300,400,300,3), np.uint8)
	for i in range(3):
		norain_img[i*100:(i+1)*100,:,:,:], rain_img[i*100:(i+1)*100,:,:,:] = syn_test(store_syn_img=True)
	save_input_path= os.path.join('D:\\CCTV\\rainfallcamera\\PReNet\\datasets', 'train_input.h5')
	save_target_path= os.path.join('D:\\CCTV\\rainfallcamera\\PReNet\\datasets', 'train_target.h5')
	target_h5f= h5py.File(save_target_path,'w')
	input_h5f= h5py.File(save_input_path, 'w')
	train_num=0

	for i in range(len(rain_img)):
		norain_img= np.float32(normalize(norain_img))
		rain_img= np.float32(normalize(rain_img))
		input_patches= Im2Patch(rain_img[i,:,:,:].transpose(2,0,1), win, 80)
		target_patches= Im2Patch(norain_img[i,:,:,:].transpose(2,0,1), win, 80)

		for n in range(target_patches.shape[-1]):
			target_data = target_patches[:, :, :, n].copy()
			target_h5f.create_dataset(str(train_num), data=target_data)

			input_data = input_patches[:, :, :, n].copy()
			input_h5f.create_dataset(str(train_num), data=input_data)

			train_num += 1

	print("total trainning samples ", train_num)
	target_h5f.close()
	input_h5f.close()

class Dataset(udata.Dataset):
	def __init__(self, data_path='.'):
		super(Dataset, self).__init__()

		self.data_path = data_path

		target_path = os.path.join(self.data_path, 'train_target.h5')
		input_path = os.path.join(self.data_path, 'train_input.h5')

		target_h5f = h5py.File(target_path, 'r')
		input_h5f = h5py.File(input_path, 'r')

		self.keys = list(target_h5f.keys())
		random.shuffle(self.keys)
		target_h5f.close()
		input_h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):

		target_path = os.path.join(self.data_path, 'train_target.h5')
		input_path = os.path.join(self.data_path, 'train_input.h5')

		target_h5f = h5py.File(target_path, 'r')
		input_h5f = h5py.File(input_path, 'r')

		key = self.keys[index]
		target = np.array(target_h5f[key])
		input = np.array(input_h5f[key])

		target_h5f.close()
		input_h5f.close()

		return torch.Tensor(input), torch.Tensor(target)

if __name__ =='__main__':
	pyh5(100)




