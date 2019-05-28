from model import RainNet
import torch.nn as nn
import torch
import cv2
from utils import autocrop
import numpy as np
import matplotlib.pyplot as plt

def test(prev, now, model_path='logs/RainNet.pth',use_gpu=True):
	model= RainNet()
	if use_gpu:
		model= model.cuda()
	model.load_state_dict(torch.load(model_path))
	img_1= cv2.imread(prev)
	# rows, cols= autocrop(img_1, window_size=(401,401))
	# img_1= img_1[rows[0]:rows[1],cols[0]:cols[1],:]
	img_gray_1= cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)/255.
	img_2= cv2.imread(now)
	# rows,cols= autocrop(img_2, window_size=(401,401))
	# img_2= img_2[rows[0]:rows[1],cols[0]:cols[1],:]
	img_gray_2= cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)/255.
	inputs= np.stack((img_gray_1, img_gray_2))
	inputs= inputs[:,np.newaxis,:,:]

	assert inputs.shape==(2,1,401,401),'shape not consistent'
	inputs= torch.Tensor(inputs)

	if use_gpu:
		inputs= inputs.cuda()

	model.eval()
	streak_1, streak_2= model(inputs)
	streak_1, streak_2= torch.clamp(streak_1,[0.,1.]),torch.clamp(streak_2,[0.,1.])
	streak_1, streak_2= np.array((streak_1.cpu().detach())*255.).astype(np.uint8),np.array((streak_2.cpu().detach())*255.).astype(np.uint8)

	streak_1, streak_2= streak_1.squeeze(), streak_2.squeeze()
	print(streak_1, streak_1.shape)
	# plt.imshow('prev',streak_1)
	# plt.imshow('now',streak_2)
	cv2.imshow('prev',streak_1)
	cv2.imshow('now',streak_2)
	cv2.imshow('prev_ori',(img_gray_1*255.).astype(np.uint8))
	cv2.imshow('now_ori',(img_gray_2*255.).astype(np.uint8))	
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__=='__main__':

	test('datasets/20180401/20180401152806.png',
		'datasets/20180401/20180401152807.png')
