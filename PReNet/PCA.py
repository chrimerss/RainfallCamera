'''
This module utilizes the morphology of rain steaks (characteristics) to detect rainfall
The pipeline goes as follows:
	1. initial rainfall location detection
		: Moving window to threshold the location of rainfall
	2. rainfall characteristics analysis with PCA decomposition
	3. refinement of rainfall streaks
	4. relocate where rainfall occurs

More info: Rain Removal By Image Quasi-Sparsity Priors
Updated 2019.3.13

Li Zhi
'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
import sys

if not sys.warnoptions:
	warnings.simplefilter('ignore')


class RainDetection_PCA():

	def __init__(self, working_dir):
		self.working_dir= working_dir

	def read_single_img(self,img_name='RainImg-3.png'):
		img_path= os.path.join(self.working_dir, img_name)
		img= cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		return img, img_name

	def init_loc(self, img, window_size=2):
		# window_size: int, seperate image into (window_size, window_size)
		h,w= img.shape
		stride= int(h/window_size)
		tot=0
		for i in range(window_size):
			for j in range(window_size):
				tot+=img[i*stride:(i+1)*stride, j*stride:(j+1)*stride].mean()
		return tot/window_size/window_size

	def _connected_components(self, img):
		return cv2.connectedComponents(img, connectivity=8)

	def _location_vector(self, ret, labels):

		#ret, labels= _connected_components(img)
		h,w= labels.shape
		X=[]
		Y=[]
		for label in np.unique(labels):
			X_series,Y_series = np.where(labels==label)
			X.append(X_series.mean())
			Y.append(Y_series.mean())
		return np.array(X), np.array(Y)

	def _covariance(self, labels, X, Y, c):
    	#labels= connected components
    	#X,Y= _location_vector()
		L,W,THETA= [],[],[]
		for i,label in enumerate(np.unique(labels)):
			X_series,Y_series= np.where(labels==label)
			mat= np.zeros((2,2))
			for x,y in zip(X_series, Y_series):
				mat= np.dot(np.array([[x],[y]]),np.array([[x,y]]))+mat
			mat/=len(X_series)
			comat= mat-np.dot(np.array([[X[i]],[Y[i]]]),np.array([[X[i], Y[i]]]))
			w,v= np.linalg.eig(comat)
			theta= np.arctan(v[0][1]/v[0][0])
			l= max(c*w)
			w= min(c*w)
			L.append(l)
			W.append(w)
			THETA.append(theta)

		return np.array(L), np.array(W), np.array(THETA)


	def _refinement(self, L, W, theta, max_width=1, l_w_ratio=10, abs_angle=45):
    	# final refinement of rain streaks
    	# return indexs
		indexs= np.where((W<max_width) &(W>0) & (L>0) & (L/W>l_w_ratio) \
    						& (abs(theta)<abs_angle))[0]
		return indexs

	def _relocate(self, indexs, labels):
    	# indexs= _refinement()
    	# labels= connectedcomponent()
		FIRST= True
		for label in indexs:
			X, Y= np.where(labels== label)
			if FIRST:
				orig_X= np.array(X)
				orig_Y= np.array(Y)
				FIRST=False
			else:
				orig_X= np.r_[orig_X, X]
				orig_Y= np.r_[orig_Y, Y]
		return orig_X, orig_Y

	def _threshold(self, img, threshold):
		img[img>threshold]= 255
		img[img<threshold]= 0
		# bi_img= cv2.bilateralFilter(img,9,75,75)
		# new_img= img-bi_img
		# new_img[new_img>200]=255
		# new_img[new_img<200]= 0
		return img

	def execute(self,img_name, connection_show=True, save_img=True, show_img=True):
		img, name= self.read_single_img(img_name)
		start= time.time()
		print('read iamge: ',name,'\nstart processing...')
		val= self.init_loc(img)
		img= self._threshold(img, val)
		ret, labels= self._connected_components(img)
		if connection_show==True:
			self.imshow_components(labels)
		X,Y= self._location_vector(ret, labels)
		covariances= self._covariance(labels, X, Y)
		L, W, THETA= self._eigtn_char(covariances, 0.5)
		indexs= self._refinement(L, W, THETA, max_width=0.5, l_w_ratio=10, abs_angle=30)
		orig_X, orig_Y= self._relocate(indexs, labels)
		img_new= np.zeros((img.shape), dtype=np.uint8)
		img_new[orig_X, orig_Y]= 255
		end= time.time()
		print('finished... \ntotal elapsed time: ', round(end-start, 2), 'seconds')
		if save_img==True:
			self.save_img(img_new, name)
		if show_img==True:
			cv2.imshow('original image', img)
			cv2.imshow('rainstreaks', img_new)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		return img_new

	def gray_img_derain(self, image, connection_show=False, show_img=True):
		#Single gray-scale image derain!
		img, name= self.read_single_img(img_name= image)
		ret, labels= self._connected_components(img)
		if connection_show==True:
			self.imshow_components(labels)
		X,Y= self._location_vector(ret,labels)
		L, W, THETA= self._covariance(labels, X, Y, 0.5)
		indexs= self._refinement(L, W, THETA, max_width=0.5, l_w_ratio=10, abs_angle=30)
		orig_X, orig_Y= self._relocate(indexs, labels)
		img_new= np.zeros((img.shape), dtype=np.uint8)
		img_new[orig_X, orig_Y]= 255
		if show_img==True:
			cv2.imshow('original', img)
			cv2.imshow('streaks', img_new)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		return img_new

	def gray_frame_derain(self,frame):
		# single frame derain, for RNN
		ret, labels= self._connected_components(frame)
		X,Y= self._location_vector(ret,labels)
		L, W, THETA= self._covariance(labels, X, Y, 0.5)
		indexs= self._refinement(L, W, THETA, max_width=0.5, l_w_ratio=10, abs_angle=30)
		orig_X, orig_Y= self._relocate(indexs, labels)
		img_new= np.zeros((frame.shape), dtype=np.uint8)
		img_new[orig_X, orig_Y]= 255

		return img_new

	def save_img(self, img, name):
		cv2.imwrite('RainStreak-'+f'{name}.png', img)


	def imshow_components(self,labels):
    # Map component labels to hue val
		label_hue = np.uint8(179*labels/np.max(labels))
		blank_ch = 255*np.ones_like(label_hue)
		labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
		labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
		labeled_img[label_hue==0] = 0
		cv2.imshow('colored connetion', labeled_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ =='__main__':
	# work_dir= './Rainy'
	img_path= 'D:\\Radar Projects\\lizhi\\CCTV\\Test_imgs\\20180324-0307\\Rain-6.png'
	img=cv2.imread(img_path)
	img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print(img)
	# work_dir= os.getcwd()
	RD= RainDetection_PCA(img_path)
	# RD.execute('Heavy-rain.png',connection_show=False, save_img=True, show_img=False)
	img_new= RD.gray_frame_derain(img)
	cv2.imshow('output', img_new)
	cv2.waitKey(0)
	cv2.destroyAllWindows()