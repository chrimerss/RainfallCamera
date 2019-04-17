'''
Basic Description:
---------------
This Project "Rainfall Camera" simply takes the rainy image or video as input, and returns rainfall intensity as output.
More info see README.md

To-do List:
---------------
[] add auto-cropping in method 'autocrop'
[] GUI design
[] retrain SVM inside classification folder with more images that no rain but before an event

Updates:
----------------
2019.04.17: events calculation with dask to return rainfall intensity, stored all images in one event.
2019.04.17: add logging to the current folder
'''
from datahandler import DataHandler
from joblib import load
import classification.classification as clf
import PReNet.PCA
import PReNet.DataPrep
import PReNet.RainProperty
from PReNet.Execute_RR_cpu import RRCal
from PReNet.generator import Generator_lstm
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
OPT= parser.parse_args()
#-------------------------------------------------------------------------------

#-------------------------Environment setup if needed---------------------------
__author__='lizhi'
__version__=0.0
if OPT.logging_file:
	logging_file= datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.log'
	logging.basicConfig(filename=logging_file, filemode= 'w', 
                    format= '%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.INFO)
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
	def __init__(self, class_model='svm_model.joblib', RNN_model='PReNet/logs/real/PReNet1.pth'):
		super(Rainfall, self).__init__()
		self.class_model= class_model
		self.RNN_model= RNN_model


	def controller(self):
		# free storage, apply crop function with xarray.DataArray.apply_ufunc or drop xarray and only stores timestamp
		self.svm= self.classfier_model()
		self.rnn= self.rnn_model()                         #to add model in advance so that reduce IO time 
		self.timeseries= {}
		first=True
		start= time.time()
		for array, timerange in self.process():
			n,h,w,c= array.shape

			sized_imgs= array.map_blocks(self.resize, chunks=(1,300,300,3), dtype=np.uint8, name='resize')
			info= sized_imgs.map_blocks(self.img_info, chunks=(1,5*900), drop_axis=[2,3],dtype=np.float32, name='information') #(-1,4500)
			self.labels= info.map_blocks(self.label, chunks=(1, ), drop_axis=1, dtype=str,name='label').compute(scheduler='threads')
			start_rain= time.time()
			array= array.map_blocks(self.crop, chunks=(1,400,300,3), dtype=np.uint8,name='crop')
			intensity= da.map_blocks(self.distribution, array, chunks=(1,), drop_axis=[1,2,3,4],
												 dtype=np.float32).compute(scheduler='threads')
			end_rain= time.time()
			print('processing rainfall costs ', round((end_rain-start_rain)/60), ' minutes')
			# timeseries[]
			if first:
				df= pd.DataFrame(columns=['Rainfall'])
				first=False

			_df= pd.DataFrame(index=timerange, columns=['Rainfall'])
			_df.Rainfall= intensity
			df= pd.concat([df, _df])
			if OPT.logging_file:
				logging.info(_df)
				logging.info(f'processing one event containing {array.shape[0]} images costs {round((end_rain-start_rain)/3600)}  hours')
		end=time.time()
		print('total elapsed time :', round((end-start)/3600), ' hours!')	
		return df

	def distribution(self, block, block_id=None):
		#input: dask array block
		#output: rainfall intensity
		img= block.squeeze().copy()
		label= self.labels[block_id[0]]
		print(label)
		if label== 'normal':
			intensity= self.normal(img)
		elif label=='night':
			intensity= self.night(img)
		elif label=='no rain':
			intensity= self.norain(img)
		elif label=='heavy':
			intensity= self.heavy(img)

		return np.array(intensity)[np.newaxis]

	def classfier_model(self):
		svm= load(self.class_model)

		return svm

	def rnn_model(self, model_path= './PReNet/logs/real/PReNet1.pth', recur_iter=4 ,use_GPU= False):
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

	def normal(self, src):
		# rainfall calculation under normal condition
		return RRCal().img_based_im(src, self.rnn)

	def heavy(self, src):
		# heavy rainfall regression model adds here
		return np.nan

	def norain(self, src):
		return 0

	def night(self, src):
		# night rainfall calculation adds here
		return np.nan

	def img_info(self, block):
		return clf.Classifier(window_size=(10,10),
							whether_client=False,
							workers=8,
							verbose=False,
							grid_search=False,
							model_path=None).img_information(block)

	def resize(self,block):
		img= block.squeeze().copy()

		return cv2.resize(img, (300,300))[np.newaxis, np.newaxis,:,:,:]

	def auto_crop(self):
		pass

	def crop(self, block, crop_window=(slice(600,1000),slice(300,600))):
		img= block.squeeze().copy()[crop_window[0], crop_window[1], :]

		return img[np.newaxis,...]



if __name__ =='__main__':
	rainfall= Rainfall().controller()
	rainfall.to_csv('rainfall.csv')