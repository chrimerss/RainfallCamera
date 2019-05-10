import requests
import cv2
import numpy as np
import scipy.signal
import time
from numba import jit
import sys
import datetime
import torch
from joblib import load
import twilio
from twilio.rest import Client
#add rainfallcamera module
sys.path.append('../rainfallcamera/')
from PReNet.generator import Generator_lstm
from PReNet.execute_cpu import RRCal

#SMS configuration, pls comment if you do not want to use SMS alert
client= Client('ACed145775d56edb163f7c8790e8947fcf', '66f63f8d7ee7194804ed0e0b06c978a1')

def get_video(use_GPU, ip='10.65.1.71',port='8080',username='admin', password='hydro254'):
	#construct url
	url= 'http://%s:%s@%s:%s/stream/video/mjpeg?resolution=HD' %(username, password, ip, port)
	cap= cv2.VideoCapture(url)
	first= True
	rnn_model= readmodel(use_GPU)
	cls_model= class_model()
	records_time= datetime.datetime.now().strftime("%Y-%m-%d")
	first_record=True
	while True:
		ret, frame= cap.read()
		# crop for the first time:
		if first:
			start_crop=time.time()
			rows,cols= autocrop(frame)
			first=False
			end_crop= time.time()
			print('processing crop uses %d seconds!' %(end_crop-start_crop))
		values= cal_values(frame)
		label= cls_model.predict(values)
		frame= drawrec(frame,rec=(cols, rows))
		localtime= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		if label=='normal':
			rainrate= raindetect(frame[rows[0]:rows[1],cols[0]:cols[1]], rnn_model, use_GPU)
			text= f'rainfall rate: {round(rainrate,2)} mm/hour'
			frame= drawtext(frame, text,(0,60))
			with open(f'records-{records_time}.txt','w+') as f:
				f.write(f'{localtime},{rainrate},')
		elif label=='no rain':
			rainrate=0
			text= f'rainfall rate: {round(rainrate,2)} mm/hour'
			frame= drawtext(frame, text,(0,60))		
		elif label=='heavy':
			frame= drawtext(frame, 'rainfall rate: unknown',(60,0))
			if first_record:
				msg= client.messages.create(
						body='Hey allen, it seems heavy rainfall is coming, remember to record videos!!!\nYours.',
						to='+6584558493',
						from_='+15109240054'
										) #send SMS msgs
				first_record=False
		else:
			frame= drawtext(frame, 'rainfall rate: unknown',(10,0))
		frame= drawtext(frame, localtime, (0,40))
		frame= cv2.putText(frame, 'REAL-TIME RAINFALL MONITORING', (0,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,(0,0,255))
		cv2.imshow('test', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
	cv2.destroyAllWindows()

	return None

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

def readmodel(use_GPU, model_path= '../rainfallcamera/PReNet/logs/real/PReNet1.pth'):
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

def drawtext(frame, text, location):
	if not isinstance(text, str):
		raise ValueError('input text is not str!')
	font = cv2.FONT_HERSHEY_DUPLEX
	frame= cv2.putText(frame, text, location, font, 0.5, (0,0,255))

	return frame

def class_model(model_path='../rainfallcamera/classification/svm_model-4-grid_searched-200x200.joblib'):
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

if __name__=='__main__':
	get_video(use_GPU=True)