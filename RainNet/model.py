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

class ContextualLayer(nn.Module):
	def __init__(self,input_channels,hidden_channels,out_channels,kernel_size=5):
		super(ContextualLayer, self).__init__()
		self.input_channels= input_channels
		self.hidden_channels= hidden_channels
		self.out_channels= out_channels
		self.kernel_size= kernel_size

		self.p1_en= nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, stride=1, padding=2, dilation=1) #(399,399)
		self.p2_en= nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, stride=1, padding=4, dilation=2) 
		self.p3_en= nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, stride=1, padding=8, dilation=3) 
		self.p1_de= nn.ConvTranspose2d(self.hidden_channels, self.out_channels,self.kernel_size,stride=1,padding=2,dilation=1) #(399,399)
		self.p2_de= nn.ConvTranspose2d(self.hidden_channels, self.out_channels, self.kernel_size, stride=1, padding=4,dilation=2) #(401,401)
		self.p3_de= nn.ConvTranspose2d(self.hidden_channels, self.out_channels, self.kernel_size, stride=1, padding=8,dilation=3)

		self.p1= nn.Sequential(self.p1_en,
						nn.ReLU(True),
						self.p1_de,
						nn.ReLU(True))
		self.p2= nn.Sequential(self.p2_en,
						nn.ReLU(True),
						self.p2_de,
						nn.ReLU(True))
		self.p3= nn.Sequential(self.p3_en,
						nn.ReLU(True),
						self.p3_de,
						nn.ReLU(True))

	def forward(self, x):
		out1= self.p1(x)
		out2= self.p2(x)
		out3= self.p3(x)

		out= torch.cat([out1, out2, out3], dim=1)

		return out


class RainNet(nn.Module):
	def __init__(self, bsize, tsize,use_gpu=True):
		super(RainNet,self).__init__()
		self.bsize=bsize
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
		self.featurenet_3= nn.Sequential(
			nn.Conv2d(3,32,3,stride=1,padding=1),		   #(32,401,401)
			nn.MaxPool2d(2),                               #(32,200,200)
			nn.ReLU(True),
			nn.Conv2d(32,64,3,1,1),							#(32,100,100)
			nn.MaxPool2d(2),
			nn.ReLU(True),
			nn.Conv2d(64,32,3,1,1),
			nn.Upsample(scale_factor=2, mode='nearest'),    #(32,200,200)
			nn.ReLU(True),
			nn.Conv2d(32,16,3,1,1),
			nn.Upsample(scale_factor=2, mode='nearest'),    #(16,400,400)
			nn.ReLU(True),
			nn.Conv2d(16,1,4,padding=2,stride=1),
			nn.ReLU(True)
			)
		self.featurenet_1= nn.Sequential(
			nn.Conv2d(1,32,3,stride=1,padding=1),		   #(32,401,401)
			nn.MaxPool2d(2),                               #(32,201,201)
			nn.ReLU(True),
			nn.Conv2d(32,64,3,1,1),							#(32,101,101)
			nn.MaxPool2d(2),
			nn.ReLU(True),
			nn.Conv2d(64,32,3,1,1),
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ReLU(True),
			nn.Conv2d(32,16,3,1,1),
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.ReLU(True),
			nn.Conv2d(16,1,4,padding=2,stride=1),
			nn.ReLU(True)
			)
		self.contexual= ContextualLayer(1,32,1)


	def forward(self,x):
		# high-pass filter

		# low-pass filter

		# x size (bsize, 2, 1, 401,401)
		for it in range(tsize-1):
			x_prev= x[:,tsize,:,:,:]
			x_now= x[:,tsize+1,:,:,:]
			out_prev= self.contexual(x_prev)

		x_prev= x[:,0,:,:,:]
		x_now= x[:,1,:,:,:]

		out_prev= self.contexual(x_prev)  #(bsize,3,401,401)
		out_now= self.contexual(x_now)    #(bsize,3,401,401)

		# print(out_prev.size())
		out_prev= self.featurenet_3(out_prev)  #(bsize,1,401,401)
		out_now= self.featurenet_3(out_now)    #(bsize,1,401,401)
		# print(out_prev.size())
		# rain feature extraction
		mask_prev,_= self.thresholding(out_prev)
		mask_now,_ = self.thresholding(out_now) 

		if self.use_gpu:
			mask_prev, mask_now= mask_prev.cuda(), mask_now.cuda()
		# print(mask_prev.size())
		# concatenaet
		out_prev= torch.cat([out_prev, mask_prev], dim=1)   #(bsize, 2, 401,401)
		out_now= torch.cat([out_now, mask_now], dim=1)

		# print(out_prev.size())

		#downsample
		out_prev= self.downsample_net_2_1(out_prev)        #(bsize,1,401,401)
		out_now= self.downsample_net_2_1(out_now)

		out_prev_feature= self.featurenet_1(out_prev)
		out_now_feature= self.featurenet_1(out_now)        #(bsize,1,401,401)

		#concatenate
		out_prev= torch.cat([out_prev, out_prev_feature], dim=1) #(bsize,2,401,401)
		out_now= torch.cat([out_now, out_now_feature], dim=1) #(bsize,2,401,401)

		out_prev= self.downsample_net_2_1(out_prev)
		out_now= self.downsample_net_2_1(out_now)

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


class RainConv2d(nn.Conv2d):
	''' Customized kernel to filter rain streaks'''
	def __init__(self, orientation):
		super(RainConv2d,self).__init__()
		kernel= cv2.getGaborKernel(ksize=(3,3),sigma=2,theta=orientation,lambd=10,gamma=0.5,psi=0)
		kernel= Variable(kernel, requires_grad=True)

	def forward(self, x):
		pass