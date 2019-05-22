# This file indicates how to detect rain intensity for a given video matrix (h,w,n)
# fully detailed paper illustration in "Towards the camera rain gauge" ---P. Allamano etc.
# Log: 2018/11/13 can not find roots for distance. How to deal with it? 
#				  rule out diameter out of range (0.5,6) pixels or mm
#				  after calculating drop velocity, have to compute the volume and then folowing rain rate.
#Log: 2019/03/21  Drop velocity is not in line with reality.
#Log: 2019/03/28  How to deal with low resolution? The background is good but results are not reasonable.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
from scipy.optimize import fsolve, minimize
import logging
import configparser




class RainProperty:
	''' this class helps to detect some rainfall properties, finally the ultimate goal is to find out the rainfall intensity.
		for CCTV detection, the parameters could be found in https://www.quora.com/Whats-the-important-parameter-of-CCTV-camera

		Camera Para. :
		focal length: 2.8mm
		sensor height: 1/3'' equivelent to 3.6mm
		exposure time: 1/200
		focus distance: have not found supportive material yet, set it to 2.0m
		aperture diameter: f1.8 D = f/1.8
		-----------------
		Attributes:
		mat: matlab matrix read by scipy.io. 3-D array (h,w,n) (adjust)
		file_path: privide the directory for reading mat (adjust)
		focal_len: focal length (mm) for specific camera (adjust)
		ex_time: exposure time (s) for specifi camera (adjust)
		f_num: F number, calculated by (focal length)/(aperture diameter) (adjust) info: https://en.wikipedia.org/wiki/F-number 
		focus_dist: focus distance (m) for specific camera (adjust)
		sensor_h: sensor height (mm) for specific camera (adjust)
		threshold: threshold for selecting rain streak candidate (brightness) (adjust)
		del_l: delta l, maximum positive brightness impulse due to a drop; set as default 50
		A : aperture diameter (mm) for specific camera.
		h,w: the height and width (pixels) for the image.
		-----------------
		Methods:
		StreakLength(graph): compute the diameter and streaklength for blured rain streak;
			Args: 
				graph, bool. plot cited image or not
			return: 
				lengths: list; calculated streak lengths ordered in contour way
				diameters: list; ordered in contour way within the range of (0.5mm,6mm)

		CalDistance(): calculate the real distance from the lens
			Args: None
			Return:
				distances: list; ordered in contour way

		_Dist_fun(): optimization function used for CalDistance

	'''
	def __init__(self, mat,focal_len=20, ex_time=1/250.,f_num=1.85,
				focus_dist=50., sensor_h=2.8, del_l=50,threshold=0.08,streak_diameter=10,verbose=False,graph=False,):
		# pass in the video matrix (h,w,n) haven't considered the colored image yet.
		# Some camera parameters need to pass in as well. focus length, exposure time, focus distance etc.
		self.mat = mat  # mat should be a matlab mat name and associated with the named in matlab
		self.focal_len = focal_len # focal length for typical CCTV is 3.6mm
		self.ex_time = ex_time
		self.f_num= f_num
		self.focus_dist = focus_dist
		self.sensor_h = sensor_h    #info for sensor height could be found in :https://en.wikipedia.org/wiki/Image_sensor_format#Table_of_sensor_formats_and_sizes
		self.A = self.focal_len/self.f_num
		self.threshold = threshold
		self.h, self.w = self.mat.shape

		self.del_l = del_l #Assumption. how to determine it?

		condition = self.mat > self.threshold
		self.mat[~condition] = 0
		self.mat[condition] = 255
		self.diameters = []
		self.lengths =[]
		self.verbose= verbose
		self.graph= graph
		self.streak_diameter=streak_diameter
		assert self.streak_diameter<self.A, 'threshold diameter is higher than A'

	def streak_process(self,image):
		'''
		Args:
		-----------------
			image: image frame, grayscale
			graph: bool, determine show marked graph or not

		Return:
		-----------------
			dists: list, lists of distances
		'''
		image = image.astype(np.uint8).copy()

		contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, \
                            cv2.CHAIN_APPROX_SIMPLE)   # detect the contour lines
		colored_img = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
		V= []
		for contour in contours:
			M = cv2.moments(contour)
			x,y,w,h = cv2.boundingRect(contour)
			font=cv2.FONT_HERSHEY_SIMPLEX
			if w != 0 and h !=0:
				length= np.sqrt(w**2+h**2)
				area = cv2.contourArea(contour)
				diameter = self._pixel2mm(area/length)
				# cv2.rectangle(colored_img,(x,y),(x+w,y+h),(0,0,255),1)
				# cv2.putText(colored_img,f'{int(length)}', (x,y),font,0.2,(0,0,255),1,cv2.LINE_AA)
				if diameter<self.streak_diameter:
					self.diameters.append(diameter)
					self.lengths.append(length)
					dist=self.cal_distance(length, diameter)
					V.append(self.cal_drop_v(diameter, dist))
					cv2.rectangle(colored_img,(x,y),(x+w,y+h),(0,0,255),1)
					cv2.putText(colored_img,f'{int(length)}', (x,y),font,0.2,(0,0,255),1,cv2.LINE_AA)				# print(diameter)
			
		if self.graph == True:
			cv2.imshow('detected rain streak',colored_img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return V

	def cal_distance(self, length, diameter, silent=True):
		# Note some are None values. How to deal with it?
		#Issue: can not find root for quadradic equations.
		diameters=np.asarray(self.diameters)
		# produce the first constrain: 1. volume within the (2/3z0, 2z0) has a blured diameter smaller than A/2
		length = self._pixel2mm(length)
				# print('length',length)

		if diameter< self.A:
			alpha = (length-diameter)/self.ex_time/1000*self.sensor_h/self.focal_len/self.h
			beta = 21.62*diameter*self.sensor_h/self.focal_len/self.h
			gamma = 21.62*self.A/1000/self.focus_dist*(1-2*self.threshold/self.del_l)
			dist = fsolve(self._dist_fun, x0=np.array([0.0,2.0]), args=(self.focus_dist, alpha, beta, gamma)) # typically two roots
			if beta**2-4*gamma*alpha<0 or len(dist)==2:
				if not silent:
					print(f"warning: function cannot find roots dealing with diameter {round(diameter,2)}")
				# the second constrain; 2. we assign a probability to see whether it falls before z0 or after z0
				dist = [dist[0] if np.random.uniform()>0.5 else dist[1]]

				# diameter_pool.append(diameter)
		# if len(self.diameters)!= len(dists): self.diameters= diameters[diameters==diameter_pool]
		if not silent:
			print(f"{len(dists)} of rain drops fall inside the control volume")
		
		return dist # this diameter and dists are in IS standard
	
	def _dist_fun(self,z,z0,alpha, beta, gamma):
		# used for optimizing funciton
		return alpha**2*z**2-beta*z+gamma*abs(z-z0)

	def _pixel2mm(self, pix):
		# this converts pixel to mm
		#focal lens equation: 1/f=1/d1+1/d2
		#The focus object is the tree which is about 50m away
		d2= 50*1000
		d1= self.focal_len*d2/(d2-self.focal_len)
		h_mm= self.focal_len/d1*3000
		pixel2mm= h_mm/self.h
		# print('1 pixel equals',pixel2mm,'mm')
		return pixel2mm*pix

	def control_volumn(self):
		# this calculates the total control volume in the range (2/3z0, 2z0)

		return 52/81*(self.focus_dist*self._pixel2mm(self.w)/self._pixel2mm(self.h)*4*(self._pixel2mm(self.h)/1000)**2)

	def cal_drop_v(self, diameter, dist):
		# this method calculates drop velocity for each rain drop candidate.
		try:
			d_p = self.sensor_h/self.focal_len/self._pixel2mm(self.h)*np.array(dist)*1000
			v = np.sqrt(21.62*diameter*d_p)
				# print('diameter:',diameter,'d_p: ', d_p, 'distance: ',dist)
			return v
		except ValueError:
			print('distance cannot be calculated ', dist,f'\nsensor_h: {self.sensor_h}, focal length: {self.focal_len} h: {self.h}')
		# print(np.asarray(V).max())
		

	def rainrate(self):
		# This calculates rain rate with a control volume approach
		# dimension mm/h
		V = self.streak_process(self.mat)
		assert len(V)==len(self.diameters), f'length of V is {len(V)}, and diameter {len(self.diameters)}'
		total_rain_rate = (1/6*np.pi*np.asarray(self.diameters)**3*np.asarray(V)*3.6*10**(-3)/self.control_volumn()).sum()
		if self.verbose:
			print('rain:  ',round(total_rain_rate,2), '   Velocity:', round(np.asarray(V).max(),2), '    diameters: ', round(np.asarray(self.diameters).max(),2))
		return total_rain_rate


# if __name__=='__main__':
# 	img_path= 'D:\\Radar Projects\\lizhi\\CCTV\\Test_imgs\\20180324-0307\\Rain-6.png'
# 	img= cv2.imread(img_path)
# 	img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	rain= RainProperty(img)
# 	print(rain.rainrate())