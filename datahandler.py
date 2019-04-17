'''
xarray and dask backend data processing pipelines
'''

import dask.array as da
from dask import delayed
import pandas as pd
import xarray as xr
import cv2
import os
import attr
from glob import glob
import datetime
import numpy as np

__author__= 'lizhi'
__version__= 0.0


class DataHandler(object):
	
	def __init__(self, base_dir='videos', folders=None):
		self.base_dir= base_dir
		self.folders=folders
		if not self.folders:
			self.folders= os.listdir(base_dir)


	def process(self):
		for folder in self.folders:
			print(f'start processing {folder} ...')
			videos= glob(os.path.join(self.base_dir, folder,'*.mkv'))
			# print(videos)
			yield self.dask_arr(videos)
			

	def _r_timestamp(self, video_path):
		#used for returning timestamp for given video file
		if not os.path.isfile(video_path):
			raise FileNotFoundError(f'the video file {video_path} provided is not correct.')
		file_name= video_path.split('/')[-1] 
		date= file_name.split('.')[0]
		date_form= datetime.datetime.strptime(date.split('_')[0]+date.split('_')[1],'%Y%m%d%H%M%S')

		return date_form

	@delayed
	def read_video(self, video):
		video= cv2.VideoCapture(video)
		frames= []
		first=True
		ind=-1
		while True:
			ret, frame= video.read()
			if not ret:
				break
			self._img_assertion(frame)
			frames.append(frame)
			ind+=1

		frames=np.array(frames)

		return frames #(h,w,c,n)

	def dask_arr(self, videos, freq='S'):
		'''xararry representation of all videos in one folder (event)'''
		start_time= self._r_timestamp(videos[0])
		lazy= [self.read_video(video) for video in videos]
		sample= lazy[0].compute()
		_,h,w,c= sample.shape
		da_array= [da.from_delayed(arr, dtype=np.uint8, shape=sample.shape) for arr in lazy]
		da_array= da.stack(da_array)
		# da_array= da.reshape(da_array, (da_array.shape[0]*da_array.shape[1], 1080,1920,3), chunks=(1,1080,1920,3))
		da_array= da_array.reshape(da_array.shape[0]*da_array.shape[1], h,w,c)
		da_array= da.rechunk(da_array,(1,h,w,c))
		print(da_array)
		end_time= start_time+ datetime.timedelta(seconds= da_array.shape[0]-1)

		return da_array, pd.date_range(start_time, end_time, freq='S')

	def _time_assertion(slef, time):
		# if not isinstance(time, )
		pass

	def _img_assertion(self, src):
		if len(src.shape)!= 3 or not isinstance(src, np.ndarray):
			raise ValueError(f'image provided {src.shape} is not correct')

if __name__=='__main__':
	base_dir= 'videos'
	folder= '20181211'
	videos= os.listdir(os.path.join(base_dir, folder))
	videos_path= [os.path.join(base_dir,folder, video) for video in videos]
	print(videos_path)
	DataHandler().xarray_repr(videos_path)
