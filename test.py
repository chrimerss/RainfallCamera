" This file calculates videos from Shijie's paper"
import cv2
from joblib import load
import datetime
from numba import jit
import numpy as np
from PReNet.generator import Generator_lstm
from PReNet.execute_cpu import RRCal
import torch
import pandas as pd


def autocrop(src, window_size=(300,300)):
	#with numba, the cost reduces from 92 seconds to 29 seconds.
	@jit(nopython=True)
	def moving_window(h,w, min_val, window_size):
    # val= overdetection(src, 2)
    # src[src<=val]=0
    # src[src>val]=255
		for i in range(h-window_size[0]):
			for j in range(w-window_size[1]):
				tot= src[i:window_size[0]+i, j:j+window_size[1]].sum()
				if tot<min_val:
					min_val= tot
					rows= (i, window_size[0]+i)
					cols= (j,j+window_size[1])

		return rows, cols

	src= cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	h,w= src.shape
	min_val= np.inf

	rows, cols= moving_window(h,w,min_val,window_size)

	return rows, cols

def class_model(model_path='classification/svm_model-4-grid_searched-200x200.joblib'):
	model= load(model_path)

	return model

def cal_values(src,window_size=(10,10)):
	frame= src.copy()
	frame= cv2.resize(frame, (200,200))
	h,w,_ = frame.shape
	n_h, n_w= h//window_size[0], w//window_size[1]
	values= np.zeros((n_h*n_w, 5), dtype=np.float32)
	ind=0
	for i in range(n_h):
		for j in range(n_w):
			sub_img= frame[i*window_size[0]: (i+1)*window_size[0], j*window_size[1]: (j+1)*window_size[1],:]
			ii,jj,mm= sub_img.shape
			err=0
			for m in range(mm):
				err+= (sub_img[:,:,m]-sub_img.mean(axis=2))**2
			values[ind,0]= err.sum()/ii/jj/mm
			values[ind,1]= sub_img.min()
			values[ind,2]= cv2.Laplacian(sub_img,cv2.CV_64F).var()
			values[ind,3]= cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)[:,:,0].mean()
			values[ind,4]= cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)[:,:,1].mean()
			ind+=1

	return values.reshape(1,-1)

def readmodel(use_GPU, model_path= 'PReNet/logs/real/PReNet2.pth'):
	model= Generator_lstm(recurrent_iter=4, use_GPU=use_GPU)
	model.load_state_dict(torch.load(model_path, map_location='cpu'))
	model.eval()
	if use_GPU:
		model= model.cuda()
	return model

def raindetect(frame, model, use_GPU):

	return RRCal().img_based_im(frame, model,use_GPU=use_GPU)

def drawrec(frame, rec):
	# frame captured by camera
	pt1= (rec[0][0], rec[1][0])
	pt2= (rec[0][1], rec[1][1])
	frame= cv2.rectangle(frame, pt1, pt2, color=(255,0,0))

	return frame

def main():
	event_name= 'Event 5'
	use_GPU= True
	cap= cv2.VideoCapture('D:/CCTV/rainfallcamera/videos/Event/%s.avi'%event_name)
	start_time= datetime.datetime(2018,7,24,11,24,43)
	first= True
	show_img=False
	ind=0
	rnn_model= readmodel(use_GPU)
	cls_model= class_model()
	df= {}
	while True:
		time= datetime.timedelta(seconds=ind)+ start_time
		ret, frame= cap.read()
		if not ret:
			break
		if first:
			rows, cols= autocrop(frame)
			first= False
		values= cal_values(frame)
		label= cls_model.predict(values)[0]
		if label=='no rain':
			rainrate=0
		elif label=='normal':
			rainrate, streak= raindetect(frame[rows[0]:rows[1],cols[0]:cols[1]], rnn_model, use_GPU)
			frame= drawrec(frame,rec=(cols, rows))
			# cv2.imshow('original',frame)
			# cv2.imshow('rainstreak', streak)
			# cv2.waitKey(0)
		elif label=='heavy':
			rainrate= np.nan
		# with open(f'{event_name}.txt','a+') as f:
		# 	f.write(f'{time},{rainrate},')		
		if show_img:
			frame= cv2.putText(frame, label, (0,40), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(0,0,255))
			frame= cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (0,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(0,0,255))
			cv2.imshow('original', frame)
			cv2.waitKey(0)

		ind+=1
		df[time]= rainrate
	pd.DataFrame.from_dict(df, orient='index').to_excel('%s.xlsx'%event_name)

if __name__=='__main__':
	main()