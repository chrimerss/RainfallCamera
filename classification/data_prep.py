'''
This Module generates datasets captured from video to classify single image into [rainy, no-rain, night, heavy-rain]
'''
import cv2
import os
import numpy as np
from glob import glob
import datetime


class GenerateData:
	def __init__(self, num_imgs=150):
		self.num_imgs= num_imgs

	def _read_video(self, video, save_dir, random_nums):
		cap= cv2.VideoCapture(video)
		ind=0
		video_name=video.split(os.sep)[-1].split('.')[0]
		date, time,_ = video_name.split('_')
		start_date= datetime.datetime.strptime(date+time, "%Y%m%d%H%M%S")
		num_imgs= len(glob(save_dir+'*.png'))
		print(num_imgs)
		while True:
			ret, frame= cap.read()
			save_date= start_date+ datetime.timedelta(seconds=ind)
			if not ret or num_imgs>=150:
				break
			elif ind in random_nums:
				print('save one image ...')
				num_imgs+=1
				cv2.imwrite(f'{save_dir+save_date.strftime("%Y%m%d%H%M%S")}.png', frame)
			ind+=1

			

	def no_rain(self, dir='datasets/no_rain/'):
		videos= glob(dir+'*.mkv')
		print(videos)
		for video in videos:
			nums= np.random.randint(1,309,size=10)
			self._read_video(video, save_dir=dir, random_nums=nums)


	def normal(self, dir='datasets/normal/'):
		videos= glob(dir+'*.mkv')
		print(videos)
		for video in videos:
			nums= np.random.randint(1,309,size=10)
			self._read_video(video, save_dir=dir, random_nums=nums)

	def heavy(self, dir='datasets/heavy/'):
		videos= glob(dir+'*.mkv')
		print(videos)
		for video in videos:
			nums= np.random.randint(1,309,size=25)
			self._read_video(video, save_dir=dir, random_nums=nums)

	def night(self,dir='datasets/night/'):
		videos= glob(dir+'*.mkv')
		print(videos)
		for video in videos:
			nums= np.random.randint(1,309,size=100)
			self._read_video(video, save_dir=dir, random_nums=nums)

	def additional(self, video, start, end, save_dir):
		video_name=video.split(os.sep)[-1].split('.')[0]
		date, time,_ = video_name.split('_')
		start_date= datetime.datetime.strptime(date+time, "%Y%m%d%H%M%S")
		cap= cv2.VideoCapture(video)
		ind=0
		while True:
			ret, frame= cap.read()
			save_date= start_date+ datetime.timedelta(seconds=ind)
			if not ret:
				break
			elif end>=ind>=start:
				if ind%15==0:
					cv2.imwrite(os.path.join(save_dir,save_date.strftime("%Y%m%d%H%M%S")+'.png'), frame)
			ind+=1

if __name__=='__main__':
	# video= 'D:\\CCTV\\RainfallCamera\\videos\\20181211\\20181211_144141_8FB5.mkv'
	# GenerateData().additional(video,0,60, 'datasets/no_rain/')
	GenerateData().normal()