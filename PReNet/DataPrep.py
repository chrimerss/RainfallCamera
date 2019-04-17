import cv2
import numpy as np
import matplotlib.pyplot as plt
import dateutil

def read_video(video_name):
	# where videos are stored:
   # change if video stores somewhere else
	video= cv2.VideoCapture(video_name)
	return video

def video2image(num_frames, video_name, store_img=True,rows=slice(600,1000), cols=slice(300,600)):
	video= read_video(video_name)
	Frames= []
	ind=0
	while True:
		ret, frame= video.read()
		if ind==num_frames or not ret:
			break
		Frames.append(frame[rows,cols])

	Frames= np.array(Frames)

	return Frames.transpose(1,2,3,0)