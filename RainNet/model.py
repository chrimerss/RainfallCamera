import torch.nn as nn
import torch
import torch.nn.functional as F
'''
Model parameters: 94505 -2019.5.26
'''

class RainNet(nn.Module):
	def __init__(self,bsize, use_gpu=True):
		super(RainNet,self).__init__()
		self.bsize=bsize
		self.use_gpu= use_gpu
		self.kernel_v= torch.Tensor([[-1,0,1],[-1,2,1],[-1,0,1]]).view(1,1,3,3)
		self.kernel_h= torch.Tensor([[-1,-1,-1],[0,2,0],[1,1,1]]).view(1,1,3,3)
		self.net= nn.Sequential(
			nn.Conv2d(1,16,3,stride=2,padding=1),		   #(16,201,201)
			nn.BatchNorm2d(16),
			nn.MaxPool2d(2, padding=1),					   #(16,101,101)
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(16,16,3,stride=2,padding=1),#(16,201,201)
			nn.BatchNorm2d(16),
			nn.Conv2d(16,32,3,1,1),							#(32,201,201)
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(32),
			nn.Conv2d(32,16,3,1,1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(16),
			nn.LeakyReLU(0.1, inplace=True),
			nn.ConvTranspose2d(16,16,3,stride=2,padding=1),#(16,401,401)
			nn.BatchNorm2d(16),
			nn.Conv2d(16,1,3,stride=1, padding=1),
			nn.ReLU(inplace=True)          #(1,401,401)
			)
		self.noise= nn.Sequential(
			nn.BatchNorm2d(2),
			nn.LeakyReLU(0.1),
			nn.Conv2d(2,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.ReLU(0.1),
			nn.Conv2d(16,32,1,1),
			nn.BatchNorm2d(32),
			nn.Conv2d(32,64,3,1,1),
			nn.BatchNorm2d(64),
			nn.Conv2d(64,64,3,1,1),
			nn.BatchNorm2d(64),
			nn.Conv2d(64,32,3,1,1),
			nn.BatchNorm2d(32),
			nn.Conv2d(32,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.ReLU(0.1),
			nn.Conv2d(16,2,3,1,1),
			nn.ReLU(0.1)
			)

	def forward(self,x):
		# x size (bsize, 2, 1, 401,401)
		x_prev= x[:,0,:,:,:]
		x_now= x[:,1,:,:,:]
		# print('before filter,',x_prev.size())
		if self.use_gpu:
			x_prev= x_prev.cuda()
			x_now= x_now.cuda()
			self.kernel_v= self.kernel_v.cuda()
			self.kernel_h= self.kernel_h.cuda()
		x_prev_v= F.conv2d(x_prev,self.kernel_v,stride=1,padding=1)
		x_now_v= F.conv2d(x_now,self.kernel_v,stride=1,padding=1)
		# print('after filter, ',x_prev_v.size())
		noise_prev= F.conv2d(x_prev,self.kernel_h,stride=1,padding=1)
		noise_now= F.conv2d(x_now,self.kernel_h,stride=1,padding=1)
		noise= torch.cat([noise_prev, noise_now],dim=1)
		# print(noise.size())
		rain_prev= self.net(x_prev_v)
		rain_now= self.net(x_now_v)
		noise= self.noise(noise)
		noise_prev= noise[:,0,:,:]
		noise_now= noise[:,1,:,:]
		bg_prev= x_prev- rain_prev-noise_prev
		bg_now= x_now- rain_now- noise_now

		return bg_prev, bg_now, rain_prev, rain_now