'''
Basic Description:
---------------
This Project "Rainfall Camera" simply takes the rainy image or video as input, and returns rainfall intensity as output.
More info see README.md

note: Put all videos want to process in one folder Let's say videos/

To-do List:
---------------
[x] animation
[x] add auto-cropping in method 'autocrop'
[ ] GUI design
[x] retrain SVM inside classification folder with more images that no rain but before an event and heavy events like in 20180401
[ ] add GPU for processing image
[x] optimize codes

Updates:
----------------
2019.04.22: add visualisation
2019.04.17: events calculation with dask to return rainfall intensity, stored all images in one event.
2019.04.17: add logging to the current folder
'''
from datahandler import DataHandler
from joblib import load
import dask
import classification.classification as clf
import PReNet.pca
import PReNet.dataprep
import PReNet.rainproperty
from PReNet.execute_cpu import RRCal
from PReNet.generator import Generator_lstm
from dask.distributed import Client, LocalCluster
import numpy as np
import dask.array as da
import torch
import pandas as pd
import os
import cv2
import time
import sys
import warnings
import datetime
import logging
import argparse
if not sys.warnoptions:
		warnings.simplefilter('ignore')

#-------------------------Argument for command----------------------------------
parser= argparse.ArgumentParser('settings')
parser.add_argument('--logging_file', default=True, help='whether use logging')
parser.add_argument('--use_GPU', default=True, help='whether use GPU for RNN')
OPT= parser.parse_args()
#-------------------------------------------------------------------------------

#-------------------------Environment setup if needed---------------------------
__author__='lizhi'
__version__=0.0
if OPT.logging_file:
	logging_file= 'logs/'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.log'
	logging.basicConfig(filename=logging_file, filemode= 'w', 
                    format= '%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO)
dask.config.set(scheduler='threads')
# cluster= LocalCluster()
# client= Client(cluster)
#-------------------------------------------------------------------------------


#-----------------------------------Main----------------------------------------
class Rainfall(DataHandler):
	"""
	This method is the abstract level to control all inputs and return rainfall intensity as .csv output
	The central processing relies on controller method

	To construct an instance, one can specify the events want to work on, by default, all folders inside videos are calculated.

	Methods:
	---------------------
		controller(): return a csv file containing all events need to calculate.


	"""
	def __init__(self, class_model='classification/svm_model-4-grid_searched-200x200.joblib', RNN_model='PReNet/logs/real/PReNet1.pth'):
		'''
		Args:
		-------------------
		basedir: the base folder where all videos that want to process are inside, by default videos/
		folders: specify the folders inside basedir, by default all
		'''
		super(Rainfall, self).__init__()
		self.class_model= class_model
		self.RNN_model= RNN_model
		self.size=(200,200)
		self.window_size=(10,10)
		logging.info(f'      Environment Setting:\nclassifier model: {class_model}\nRNN model: {RNN_model}\nresized image: {self.size}')


	def controller(self):
		# free storage, apply crop function with xarray.DataArray.apply_ufunc or drop xarray and only stores timestamp
		self.svm= self.classfier_model()
		self.rnn= self.rnn_model()                         #to add model in advance so that reduce IO time 
		self.timeseries= {}
		first=True
		start= time.time()
		for array, timerange in self.process():
			n,h,w,c= array.shape
			tot_grids= self.size[0]//self.window_size[0]*self.size[1]//self.window_size[1]
			sized_imgs= array.map_blocks(self.resize, chunks=(1,self.size[0],self.size[1],3), dtype=np.uint8, name='resize')
			sized_imgs= sized_imgs.rechunk((1,self.window_size[0], self.window_size[1], 3))
			info= sized_imgs.map_blocks(self.img_info, chunks=(1,tot_grids,5), drop_axis=3,dtype=np.float32, name='information') #(-1,4500)
			self.labels= info.map_blocks(self.label, chunks=(1, ), drop_axis=[1,2], dtype=str ,name='label').compute()
			start_rain= time.time()
			array= array.map_blocks(self.crop, chunks=(1,400,300,3), dtype=np.uint8,name='crop')
			intensity= da.map_blocks(self.distribution, array, chunks=(1,), drop_axis=[1,2,3,4],
												 dtype=np.float32).compute()
			end_rain= time.time()
			print('processing rainfall costs ', round((end_rain-start_rain)/60.,2), ' minutes')
			# timeseries[]
			if first:
				df= pd.DataFrame(columns=['Rainfall'])
				first=False

			_df= pd.DataFrame(index=timerange, columns=['Rainfall'])
			_df.Rainfall= intensity
			df= pd.concat([df, _df])
			if OPT.logging_file:
				logging.info(_df)
				logging.info(f'processing one event containing {array.shape[0]} images \
				 				costs {round((end_rain-start_rain)/3600.,2)}  hours')
		end=time.time()
		print('total elapsed time :', round((end-start)/3600.), ' hours!')	
		return df

	def distribution(self, block, block_id=None):
		#input: dask array block
		#output: rainfall intensity
		img= block.squeeze().copy()
		label= self.labels[block_id[0]]
		if label== 'normal':
			intensity= self.normal(img)
		elif label=='night':
			intensity= self.night(img)
		elif label=='no rain':
			intensity= self.norain(img)
		elif label=='heavy':
			intensity= self.heavy(img)
		print(intensity)
		return np.array(intensity)[np.newaxis]

	def classfier_model(self):
		svm= load(self.class_model)

		return svm

	def rnn_model(self, model_path= './PReNet/logs/real/PReNet1.pth', recur_iter=4 ,use_GPU= OPT.use_GPU):
		model= Generator_lstm(recur_iter, use_GPU)
		if use_GPU:
			model = model.cuda()
		model.load_state_dict(torch.load(model_path, map_location='cpu'))
		model.eval()

		return model

	def label(self, block):
		info= block.squeeze().copy().reshape(1,-1)
		label= self.svm.predict(info)
		
		return np.array(label)[np.newaxis]

	def normal(self, src, use_GPU= OPT.use_GPU):
		# rainfall calculation under normal condition
		return RRCal().img_based_im(src, self.rnn, use_GPU=use_GPU)

	def heavy(self, src):
		# heavy rainfall regression model adds here
		return np.nan

	def norain(self, src):
		return 0

	def night(self, src):
		# night rainfall calculation adds here
		return np.nan

	def img_info(self, block):
		sub_img= block.squeeze().copy()
		m,n,c= sub_img.shape
		assert sub_img.shape[:2]==self.window_size, f'Input image has shape {sub_img.shape[:2]},\
														 but expect shape {self.window_size}'
		values= np.zeros(5, dtype=np.float32)
		#RMSE
		err=0
		for i in range(c):
			err+= (sub_img[:,:,i]-sub_img.mean(axis=2))**2
		values[0]= err.sum()/m/n/c
		values[1]= sub_img.min()
		values[2]= cv2.Laplacian(sub_img,cv2.CV_64F).var()
		values[3]= cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)[:,:,0].mean()
		values[4]= cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)[:,:,1].mean()

		return values[np.newaxis,np.newaxis, :]

	def resize(self,block):
		img= block.squeeze().copy()

		return cv2.resize(img, self.size)[np.newaxis,:,:,:]

	def auto_crop(self, window_size=(300,300)):
		h,w = src.shape
		min_val= np.inf
		# val= overdetection(src, 2)
		# src[src<=val]=0
		# src[src>val]=255
		for i in range(h-window_size[0]):
			for j in range(w-window_size[1]):
				tot= src[i:window_size[0]+i, j:j+window_size[1]].sum()
				if tot<min_val: min_val=tot; rows=slice(i, window_size[0]+i); cols=slice(j,j+window_size[1])

		return rows, cols

	def crop(self, block, crop_window=(slice(600,1000),slice(300,600))):
		img= block.squeeze().copy()[crop_window[0], crop_window[1], :]

		return img[np.newaxis,...]



if __name__ =='__main__':
	rainfall= Rainfall().controller()
	rainfall.to_csv('rainfall.csv')