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
	def __init__(self, input_channels, hidden_channels, kernel_size, dilation):
		super(ContextBlock, self).__init__()
		self.input_channels= input_channels
		self.hidden_channels= hidden_channels
		self.kernel_size= kernel_size
		self.dilation= dilation

		padding= int(self.dilation*(self.kernel_size-1)/2)

		self.contextblock= nn.Sequential(
			nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size,padding=padding,
					stride=1, dilation=self.dilation),
			nn.LeakyReLU(),
			nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, padding=padding,
					stride=1, dilation=self.dilation)
			)

	def forward(self, x):
		out= self.contextblock(x)

		return out


class ContextualLayer(nn.Module):
	def __init__(self,input_channels,hidden_channels, kernel_size):
		super(ContextualLayer, self).__init__()
		self.input_channels= input_channels
		self.hidden_channels= hidden_channels
		self.kernel_size=kernel_size


		self.p1= ContextBlock(self.input_channels, self.hidden_channels, self.kernel_size, dilation=1)

		self.p2= ContextBlock(self.input_channels, self.hidden_channels, self.kernel_size, dilation=2)

		self.p3= ContextBlock(self.input_channels, self.hidden_channels,  self.kernel_size, dilation=3)

	def forward(self, x):
		out1= self.p1(x)
		out2= self.p2(x)
		out3= self.p3(x)

		out= out1+ out2+ out3

		return out


class RainNet(nn.Module):
	def __init__(self, bsize, tsize=2,use_gpu=True):
		super(RainNet,self).__init__()
		self.bsize=bsize
		self.tsize=tsize
		self.use_gpu= use_gpu
		self.kernel_v= torch.Tensor([[-1,0,1],[-1,2,1],[-1,0,1]]).view(1,1,3,3)
		self.kernel_h= torch.Tensor([[-1,-1,-1],[0,2,0],[1,1,1]]).view(1,1,3,3)
		self.downsample_net_4_1= nn.Sequential(
						nn.Conv2d(4,1,3,1,1),
						nn.ReLU(True),
						nn.BatchNorm2d(1)
							)
		self.downsample_net_2_1= nn.Sequential(
						nn.Conv2d(2,1,3,1,1),
						nn.ReLU(True),
						nn.BatchNorm2d(1)
							)
		# self.featurenet_3= nn.Sequential(
		# 	nn.Conv2d(3,32,3,stride=1,padding=1),		   #(32,401,401)
		# 	nn.MaxPool2d(2),                               #(32,200,200)
		# 	nn.ReLU(True),
		# 	nn.Conv2d(32,64,3,1,1),							#(32,100,100)
		# 	nn.MaxPool2d(2),
		# 	nn.ReLU(True),
		# 	nn.Conv2d(64,32,3,1,1),
		# 	nn.Upsample(scale_factor=2, mode='nearest'),    #(32,200,200)
		# 	nn.ReLU(True),
		# 	nn.Conv2d(32,16,3,1,1),
		# 	nn.Upsample(scale_factor=2, mode='nearest'),    #(16,400,400)
		# 	nn.ReLU(True),
		# 	nn.Conv2d(16,1,4,padding=2,stride=1),
		# 	nn.ReLU(True)
		# 	)
		# self.featurenet_1= nn.Sequential(
		# 	nn.Conv2d(1,32,3,stride=1,padding=1),		   #(32,401,401)
		# 	nn.MaxPool2d(2),                               #(32,201,201)
		# 	nn.ReLU(True),
		# 	nn.Conv2d(32,64,3,1,1),							#(32,101,101)
		# 	nn.MaxPool2d(2),
		# 	nn.ReLU(True),
		# 	nn.Conv2d(64,32,3,1,1),
		# 	nn.Upsample(scale_factor=2, mode='nearest'),
		# 	nn.ReLU(True),
		# 	nn.Conv2d(32,16,3,1,1),
		# 	nn.Upsample(scale_factor=2, mode='nearest'),
		# 	nn.ReLU(True),
		# 	nn.Conv2d(16,1,4,padding=2,stride=1),
		# 	nn.ReLU(True)
		# 	)
		self.contexual_3= ContextualLayer(1,32, kernel_size=3)
		self.contexual_5= ContextualLayer(1,32,kernel_size=5)
		self.contexual_7=ContextualLayer(1,32,kernel_size=7)
		self.seb= SEBlock()


	def forward(self,x):
		# high-pass filter

		# low-pass filter

		x_prev= x[:,0,:,:,:]
		x_now= x[:,1,:,:,:]

		out_prev_mae_1= self.contexual_3(x_prev)  #(bsize,1,401,401)
		out_now_mae_1= self.contexual_3(x_now)    #(bsize,1,401,401)


		out_prev_se_1= self.seb(out_prev_mae_1)
		out_now_se_1= self.seb(out_now_mae_1)

		out_prev_mae_2= self.contexual_5(out_prev_se_1)
		out_now_mae_2= self.contexual_5(out_now_se_1)

		out_prev_se_2= self.seb(out_prev_mae_2)
		out_now_se_2= self.seb(out_now_mae_2)

		out_prev_mae_3= self.contexual_7(out_prev_se_2)
		out_now_mae_3= self.contexual_7(out_now_se_2)

		out_prev_se_3= self.seb(out_prev_mae_3)
		out_now_se_3= self.seb(out_now_mae_3)

		out_prev= out_prev_se_1+ out_prev_se_2+ out_prev_se_3
		out_now= out_now_se_1+ out_now_se_2+ out_now_se_3

		# print(out_prev.size())
		# out_prev= self.featurenet_3(out_prev)  #(bsize,1,401,401)
		# out_now= self.featurenet_3(out_now)    #(bsize,1,401,401)
		# print(out_prev.size())
		# rain feature extraction
		# mask_prev,_= self.thresholding(out_prev)
		# mask_now,_ = self.thresholding(out_now) 

		# if self.use_gpu:
		# 	mask_prev, mask_now= mask_prev.cuda(), mask_now.cuda()
		# # print(mask_prev.size())
		# # concatenaet
		# out_prev= torch.cat([out_prev, mask_prev], dim=1)   #(bsize, 2, 401,401)
		# out_now= torch.cat([out_now, mask_now], dim=1)

		# print(out_prev.size())

		#downsample
		# out_prev= self.downsample_net_2_1(out_prev)        #(bsize,1,401,401)
		# out_now= self.downsample_net_2_1(out_now)

		# out_prev_feature= self.featurenet_1(out_prev)
		# out_now_feature= self.featurenet_1(out_now)        #(bsize,1,401,401)

		#concatenate
		# out_prev= torch.cat([out_prev, out_prev_feature], dim=1) #(bsize,2,401,401)
		# out_now= torch.cat([out_now, out_now_feature], dim=1) #(bsize,2,401,401)

		# out_prev= self.downsample_net_2_1(out_prev)
		# out_now= self.downsample_net_2_1(out_now)

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