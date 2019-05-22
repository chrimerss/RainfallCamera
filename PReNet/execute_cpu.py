'''
This is the implementation for calculating rainfall intensity based on the RRN
The pipeline goes as follow:
	1. background removal with RNN.
	2. PCA decomposition to rule out noise (control)
	3. calculate rain rate with Allamano algorithm
	4. assess this method with radar and gauge data.
'''

import cv2
from dataprep import video2image
import datetime
from rainproperty import RainProperty
from pca import RainDetection_PCA
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from utils import *
from generator import Generator_lstm
import time
import sys
import logging
from utils import autocrop_day
import configparser

import warnings
if not sys.warnoptions:
		warnings.simplefilter('ignore')
#====== read configurations===========
config= configparser.ConfigParser()
config.read('../camera.ini')
keys= ['focal_len', 'ex_time', 'f_num','focus_dist','sensor_h','del_l','threshold','streak_diameter']
PARAMS_CAM= {k:float(config['NC450'][k]) for k in keys}
PARAMS_RNN= config['PReNet']
# PARAMS_CAM= (dict(k, config['NC450'][k]) for k in keys)

if bool(PARAMS_RNN['use_GPU']):
	os.environ["CUDA_VISIBLE_DEVICES"] = PARAMS_RNN['gpu_id']

#logging_file= opt.save_path+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.log'
#logging.basicConfig(filename=logging_file, filemode= 'w', 
				#	format= '%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO)

class RRCal(object):
	'''
	Methods:
	--------------
	_return_frames, _return_videos: internal methods
	pretrained_model: load the pretrained model, there are two models for option: [PReNet1, PReNet2] stored in logs/real
	single_img_rain_intensity: return the rainfall intensity based on the Allamano algorithm; more info see RainProperty
	execute: aggregate all videos in the path specified and return the dictionary which like {date: intensity}
	'''
	def __init__(self, base_path=os.getcwd()):
		self.base_path= base_path

	@staticmethod
	def _return_frames(video):
		# frames= video2image(1000, video, store_img=False, rows=slice(600,1000), cols=slice(300,600))
		frames= video2image(1000, video, store_img=False)
		if frames is None:
			raise FileNotFoundError('please check input video path %s'%video)

		return frames

	def _return_videos(self, folder):
		videos= glob.glob(os.path.join(self.base_path, folder, '*.mkv'))

		return videos

	def pretrained_model(self, model_path="latest.pth"):
		'''
		Args:
		------------------
		model_path: specify which model to use, ['PReNet1.pth', 'PReNet2.pth']

		return:
		------------------
		model: torch model
		'''
		print('Loading model ...\n')
		model = Generator_lstm(int(PARAMS_RNN['recurrent_iter']), bool(PARAMS_RNN['use_GPU']))
		print_network(model)
		if bool(PARAMS_RNN['use_GPU']):
			model = model.cuda()
		model.load_state_dict(torch.load(PARAMS_RNN['model_path'], map_location='cpu'))
		model.eval()
   
		return model

	def activate_PCA(self):
		return RainDetection_PCA(self.base_path)

	def _tensor_test(self, video):
		'''
		Not recommended because it will exceed GPU memory
		'''
		model= self.pretrained_model()
		frames= RRCal._return_frames(video)[:,:,:,:5]
		print('frames shape:',frames.shape)
		new_frames= np.zeros(frames.transpose(3,2,0,1).shape, dtype=np.uint8)
		print('new_frames shape', new_frames.shape)
		h,w,c,n= frames.shape
		for i in range(n):
			frame= frames[:,:,:,i].copy()
			b, g, r = cv2.split(frame)
			y = cv2.merge([r, g, b])
			y = normalize(np.float32(y))
			new_frames[i,:,:,:]=y.transpose(2,0,1)
		new_frames= Variable(torch.Tensor(new_frames))
		if bool(PARAMS_RNN['use_GPU']):
			new_frames= new_frames.cuda()
		out, _ = model(new_frames)
		out = torch.clamp(out, 0., 1.)
		print('out shape:',out.shape)
		with torch.no_grad():
			if bool(PARAMS_RNN['use_GPU']):
				torch.cuda.synchronize()
		if bool(PARAMS_RNN['use_GPU']):
			save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
		else:
			save_out = np.uint8(255 * out.data.numpy().squeeze())
		print('save_out shape:',save_out.shape)

	def single_img_rain_intensity(self, img):
		'''
		Args:
		-------------------
		img: numpy.ndarray like gray-scale image 

		Return:
		-------------------
		rainrate: int More info related to RainProperty
		'''
		rainrate= RainProperty(mat=img, graph=False,**PARAMS_CAM)
		return rainrate.rainrate()

	@staticmethod
	def rainstreak(rainy, derain, threshold):
		rainy= cv2.cvtColor(rainy, cv2.COLOR_BGR2GRAY)
		derain= cv2.cvtColor(derain, cv2.COLOR_BGR2GRAY)
		diff= rainy- derain
		diff[derain<30]=0
		diff[diff>=threshold]= 255
		diff[diff<=threshold]= 0
    	
		return diff

	def img_based_im(self, frame,model, PCA=True,use_GPU=True):
		'''
		Args:
		--------------
		src: single image

		Returns:
		--------------
		intensity: mm/h
		'''
		morphology_detect= self.activate_PCA()
		input_img= frame.copy()
		b, g, r = cv2.split(frame)
		y = cv2.merge([r, g, b])
		y = normalize(np.float32(y))
		y = np.expand_dims(y.transpose(2, 0, 1), 0)
		y = Variable(torch.Tensor(y))
		if use_GPU:
			y = y.cuda()
		with torch.no_grad():
			if use_GPU:
				torch.cuda.synchronize()
			out, _ = model(y)
			out = torch.clamp(out, 0., 1.)
			if use_GPU:
				torch.cuda.synchronize()
		if use_GPU:
			save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
		else:
			save_out = np.uint8(255 * out.data.numpy().squeeze())
		save_out = save_out.transpose(1, 2, 0)
		r, g, b = cv2.split(save_out)
		save_out = cv2.merge([b, g, r])
		streak= RRCal.rainstreak(input_img, save_out, 30)

		if PCA:
			streak= morphology_detect.gray_frame_derain(streak)
		rate= self.single_img_rain_intensity(streak)
		
		return rate

	def video_based_im(self, video, PCA=True):
		'''
		Args:
		--------------
		video: the absolute path for one video

		Return:
		--------------
		rainrate: pandas.DataFrame
		
		updates:
		add auto crop to video instead if using stagnant frame
		'''
		model= self.pretrained_model()
		morphology_detect= self.activate_PCA()
		sts_time= video.split('\\')[-1].split('.')[0]
		date, daytime, _= sts_time.split('_')
		curr_date= datetime.datetime.strptime(date+daytime, '%Y%m%d%H%M%S')
		rainrate_series= {}
		frames= RRCal._return_frames(video)
		h,w,c,n= frames.shape
		start_time=  time.time()
		for i in range(n):
			print('processing current time: ', curr_date)
			frame= frames[:,:,:,i].copy()
			b, g, r = cv2.split(frame)
			y = cv2.merge([r, g, b])
			input_img= y.copy()
			y = normalize(np.float32(y))
			y = np.expand_dims(y.transpose(2, 0, 1), 0)
			y = Variable(torch.Tensor(y))
			if bool(PARAMS_RNN['use_GPU']):
				y = y.cuda()
			with torch.no_grad():
				if bool(PARAMS_RNN['use_GPU']):
					torch.cuda.synchronize()
				out, _ = model(y)
				out = torch.clamp(out, 0., 1.)
				if bool(PARAMS_RNN['use_GPU']):
					torch.cuda.synchronize()
			if bool(PARAMS_RNN['use_GPU']):
				save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
			else:
				save_out = np.uint8(255 * out.data.numpy().squeeze())
			save_out = save_out.transpose(1, 2, 0)
			b, g, r = cv2.split(save_out)
			save_out = cv2.merge([r, g, b])
			streak= RRCal.rainstreak(input_img, save_out, 40)
			if PCA:
				streak= morphology_detect.gray_frame_derain(streak)
			cv2.imwrite(os.path.join(self.base_path, curr_date.strftime('%Y%m%d%H%M%S')+'.png'), streak)
			rate= self.single_img_rain_intensity(streak)
			rainrate_series[curr_date]= rate
			print(curr_date.strftime('%Y-%m-%d %H:%M:%S')+':  ',rate)
			curr_date+= datetime.timedelta(seconds=1)
		end_time= time.time()
		print('Total elapsed time :', round((end_time-start_time)/60,2),'  minutes!' )
		df= pd.DataFrame.from_dict(rainrate_series, orient='index')

		return df


	def event_based_im(self, event, PCA=True):
		'''
		Args:
		----------------
		event: relateve folder path for videos stored in one event
		PCA: activate PCA decomposition for rain drop selection

		Return:
		----------------
		rainrate_series: pandas.DataFrame, time series of rain rate
		'''
		videos= self._return_videos(event)
		model= self.pretrained_model()
		morphology_detect= self.activate_PCA()
		print(videos)
		
		rainrate_series= {}
		for video in videos:
			sts_time= video.split(os.sep)[-1].split('.')[0]
			date, daytime, _= sts_time.split('_')
			curr_date= datetime.datetime.strptime(date+daytime, '%Y%m%d%H%M%S')
			frames= RRCal._return_frames(video)
			h,w,c,n= frames.shape
			start_time=  time.time()
			for i in range(n):
				print('processing current time', curr_date)
				frame= frames[:,:,:,i]
				input_img= frame.copy()
				b, g, r = cv2.split(frame)
				y = cv2.merge([r, g, b])
				y = normalize(np.float32(y))
				y = np.expand_dims(y.transpose(2, 0, 1), 0)
				y = Variable(torch.Tensor(y))
				if bool(PARAMS_RNN['use_GPU']):
					y = y.cuda()
				with torch.no_grad():
					if bool(PARAMS_RNN['use_GPU']):
						torch.cuda.synchronize()
					out, _ = model(y)
					out = torch.clamp(out, 0., 1.)
					if bool(PARAMS_RNN['use_GPU']):
						torch.cuda.synchronize()
				if bool(PARAMS_RNN['use_GPU']):
					save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
				else:
					save_out = np.uint8(255 * out.data.numpy().squeeze())
				save_out = save_out.transpose(1, 2, 0)
				r, g, b = cv2.split(save_out)
				save_out = cv2.merge([b, g, r])
				streak= RRCal.rainstreak(input_img, save_out, 30)

				if PCA:
					streak= morphology_detect.gray_frame_derain(streak)
				rate= self.single_img_rain_intensity(streak)
				print(rate)
				rainrate_series[curr_date]= rate
				curr_date+= datetime.timedelta(seconds=1)
				#logging.info(f'{curr_date}:       {rate}')
			end_time= time.time()
			print('----------------------------------\n'\
					'One video done!\n Elapsed time:', round((end_time-start_time)/60,2),'  minutes!')
		df= pd.DataFrame.from_dict(rainrate_series, orient='index')
		return df


if __name__=='__main__':
	# # rate= rate_cal.video_based_im('D:\\Radar Projects\\lizhi\\CCTV\\Rain Detection\\CSC\\MS-CSC-Rain-Streak-Removal\\20181211\\20181211_141041_3BBB.mkv')
	rate_cal= RRCal('D:\\CCTV\\RainfallCamera\\videos')
	# rate= rate_cal._tensor_test('D:\\Radar Projects\\lizhi\\CCTV\\Rain Detection\\CSC\\MS-CSC-Rain-Streak-Removal\\20181211\\20181211_141041_3BBB.mkv')
	# img=cv2.imread('D:\\Radar Projects\\lizhi\\CCTV\\Videos\\HightIntensity\\20180401\\20180401152758.png')[600:1000,300:600]
	# rate= rate_cal.img_based_im(img,
	# model=rate_cal.pretrained_model(model_path)

	# print(model_path,rate)
	video_path= 'D:\\CCTV\\rainfallcamera\\videos\\20190516\\20190516_130642_NC450.mp4'
	rate_cal.video_based_im(video_path,PCA=True).to_excel('D:\\CCTV\\rainfallcamera\\NC450.xlsx')