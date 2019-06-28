import torch.nn as nn
import torch
import torch.nn.functional as F



import torch.nn as nn
import torch
import torch.nn.functional as F
from numba import jit
import cv2
import numpy as np
'''
Model parameters: 94505 -2019.5.26

Model parameters: 23M   -2019.6.12

Model parameters: 23M+80K+65K   -2019.6.13

Equations to evaluate dimension: (L_in+2*padding-dilation*(kernel_size-1)-1)/stride+1
'''
class Flatten(nn.Module):
	'''Implementation of a flatten method'''

	def forward(self, x):
		bsize, c, m, n= x.size()
		x= x.view(-1, c*m*n)

		return x


class SEBlock(nn.Module):
	def __init__(self, input_channels=401*401, hidden_channels=100):
		super(SEBlock, self).__init__()
		self.se=nn.Sequential(
					Flatten(),
					nn.Linear(input_channels, hidden_channels),
					nn.ReLU(True),
					nn.Linear(hidden_channels, input_channels),
					nn.ReLU(True)
					)

	def forward(self, x):
		out= self.se(x)
		#out size (bsize, 401*401)
		out= torch.unsqueeze(out, dim=1)
		out= out.view(-1, 1, 401, 401)
		out= x*out

		return out

class ContextBlock(nn.Module):
	def __init__(self, input_channels, hidden_channels, kernel_size, dilation=1):
		super(ContextBlock, self).__init__()
		self.input_channels= input_channels
		self.hidden_channels= hidden_channels
		self.kernel_size= kernel_size
		self.dilation= dilation

		padding= int(self.dilation*(self.kernel_size-1)/2)

		self.contextblock= nn.Sequential(
			nn.LeakyReLU(),
			nn.BatchNorm2d(self.input_channels),
			nn.Conv2d(self.input_channels, self.hidden_channels*2, self.kernel_size,padding=padding,
					stride=1, dilation=self.dilation),
			nn.LeakyReLU(),
			nn.BatchNorm2d(self.hidden_channels*2),
			nn.Conv2d(self.hidden_channels*2, self.hidden_channels, self.kernel_size, padding=padding,
					stride=1, dilation=self.dilation),
			)

	def forward(self, x):
		out= self.contextblock(x)

		return out


class ContextualLayer(nn.Module):
	def __init__(self,input_channels,hidden_channels ):
		super(ContextualLayer, self).__init__()
		self.input_channels= input_channels
		self.hidden_channels= hidden_channels


		self.p1= ContextBlock(self.input_channels, self.hidden_channels, 3)

		self.p2= ContextBlock(self.input_channels, self.hidden_channels, 5)

		self.p3= ContextBlock(self.input_channels, self.hidden_channels, 7)

	def forward(self, x):
		out1= self.p1(x)
		out2= self.p2(x)
		out3= self.p3(x)

		# print(out1.size())
		out= torch.cat([out1, out2, out3], dim=1)    #(bsize, 3*hidden_channels)

		return out


class RainNet(nn.Module):
	def __init__(self, bsize, tsize=2,use_gpu=True):
		super(RainNet,self).__init__()
		self.bsize=bsize
		self.tsize=tsize
		self.use_gpu= use_gpu
		self.contexual_1= ContextualLayer(1,16)
		# self.seb_1= SEBlock()
		self.conv1x1= nn.Sequential(
						nn.Conv2d(48,16,1),
						nn.BatchNorm2d(16),
						nn.ReLU(True),
						nn.Conv2d(16,1,1),
						nn.ReLU(True)
						)


	def forward(self,x):
		# high-pass filter

		# low-pass filter

		x_prev= x[:,0,:,:,:].clone()
		x_now= x[:,1,:,:,:].clone()

		out_prev_mae_1= self.contexual_1(x_prev)  #(bsize,1,401,401)
		out_now_mae_1= self.contexual_1(x_now)    #(bsize,1,401,401)
		# print(out_prev_mae_1.size())

		# out_prev_se_1= self.seb_1(out_prev_mae_1)
		# out_now_se_1= self.seb_1(out_now_mae_1)

		out_prev= self.conv1x1(out_prev_mae_1)
		out_now= self.conv1x1(out_now_mae_1)


		bg_prev= x_prev- out_prev
		bg_now= x_now- out_now
		# cv2.imshow('high pass', (high_now.cpu().detach().numpy().squeeze()*255.).astype(np.uint8))
		# cv2.imshow('high pass diffy', (high_now_v.cpu().detach().numpy().squeeze()*255.).astype(np.uint8))
		# cv2.imshow('rain',(rain_now.cpu().detach().numpy().squeeze()*255.).astype(np.uint8))
		# cv2.imshow('background',(bg_now.cpu().detach().numpy().squeeze()*255.).astype(np.uint8))
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		return bg_prev, bg_now, out_prev, out_now


	def thresholding(self, mat, window_size=20): 
		bsize,_,h,w= mat.size()
		high= mat.detach().cpu()
		low= mat.detach().cpu()
		n_h, n_w= h//window_size, w//window_size
		
		@torch.jit.script
		def get_val(tensor : torch.Tensor, n_h : int, n_w : int,window_size:int):

			value= torch.zeros(1, dtype= torch.float64)
			for i in range(int(n_h)):
				for j in range(int(n_w)):
					value+= tensor[0,i*window_size:(i+1)*window_size,j*window_size:(j+1)*window_size].mean()
			return value

		for b in range(bsize):
			value= get_val(mat[b,:,:,:], n_h,n_w, window_size)/255.
			high[b,:,:,:][high[b,:,:,:]>=float(value)]=1
			high[b,:,:,:][high[b,:,:,:]<float(value)]=0
			low[b,:,:,:][low[b,:,:,:]>=float(value)]=0

		return high, low


class RainDescriminator(nn.Module):

	def __init__(self, use_gpu=True):
		super(RainDescriminator,self).__init__()
		self.clsnet= nn.Sequential(
								nn.Conv2d(1,16,3,padding=1,stride=2),  #(201,201)
								nn.MaxPool2d(2),				        #(100,100)
								nn.BatchNorm2d(16),
								nn.ReLU(True),
								nn.Conv2d(16,16,3,padding=1,stride=1),
								nn.MaxPool2d(2),					   #(50,50)
								nn.BatchNorm2d(16),
								nn.ReLU(True),
								nn.Conv2d(16,1,3,padding=1, stride=1),
								nn.MaxPool2d(2),					   #(25,25)
								nn.ReLU(True))

		self.fc=nn.Sequential(
								nn.Linear(1*25*25,120),
								nn.Linear(120,50),
								nn.Linear(50,1),
								nn.Sigmoid()
									)

	def forward(self,x):
		#input data shape (bsize, channel, 401, 401)
		out= self.clsnet(x)
		out= out.view(-1, 25*25*1)
		out= self.fc(out)


		return out


class BGDescriminator(nn.Module):
	def __init__(self, use_gpu=True):

		super(BGDescriminator, self).__init__()
		self.clsnet= nn.Sequential(
									nn.Conv2d(1,16,3,padding=1,stride=1),
									nn.MaxPool2d(2),                        #(200,200)
									nn.BatchNorm2d(16),
									nn.Conv2d(16,32,3,padding=1,stride=1),
									nn.MaxPool2d(2),
									nn.BatchNorm2d(32),
									nn.ReLU(True),         #(100,100)
									nn.Conv2d(32,16,3,padding=1, stride=1),
									nn.MaxPool2d(2),
									nn.BatchNorm2d(16),                        #(50,50)
									nn.Sigmoid(),
									nn.Conv2d(16,1,3,padding=1, stride=1),
									nn.MaxPool2d(2),
									nn.BatchNorm2d(1))                       #(25,25))

		self.fc=nn.Sequential(nn.Linear(1*25*25,100),
								nn.BatchNorm1d(100),
								nn.Dropout(0.2),
								nn.Linear(100,1),
								nn.Sigmoid()
								)

	def forward(self,x):
		#input data shape (bsize, channel, m, n)
		# print(x.size())
		out= self.clsnet(x)
		out= out.view(-1,1*25*25)
		# print(out.size())
		out= self.fc(out)

		return out