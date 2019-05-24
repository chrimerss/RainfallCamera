import torch.nn as nn
import torch
import torch.nn.functional as F

class RainNet(nn.Module):
	def __init__(self, use_gpu=True):
		super(RainNet,self).__init__()
		self.use_gpu= use_gpu
		self.kernel= torch.Tensor([[-1,0,1],[-1,2,1],[-1,0,1]]).view(1,1,3,3)
		self.net= nn.Sequential(
			nn.Conv2d(1,16,3,stride=2,padding=1),		   #(16,201,201)
			nn.BatchNorm2d(16),
			nn.MaxPool2d(2, padding=1),					   #(16,101,101)
			nn.ReLU(),
			nn.ConvTranspose2d(16,16,3,stride=2,padding=1),#(16,201,201)
			nn.ConvTranspose2d(16,16,3,stride=2,padding=1),#(16,401,401)
			nn.BatchNorm2d(16),
			nn.Conv2d(16,1,3,stride=1, padding=1),
			nn.ReLU()          #(1,401,401)
			)

	def forward(self,x):
		# x size (bsize, 2, 1, 401,401)
		x_prev= x[0,:,:,:]
		x_now= x[1,:,:,:]
		if self.use_gpu:
			x_prev= x_prev.cuda()
			x_now= x_now.cuda()
			self.kernel= self.kernel.cuda()
		x_prev_out= F.conv2d(x_prev,self.kernel,stride=1,padding=1)
		x_now_out= F.conv2d(x_now,self.kernel,stride=1,padding=1)
		x_prev_out= self.net(x_prev_out)
		x_now_out= self.net(x_now_out)
		bg_prev= x_prev- x_prev_out
		bg_now= x_now- x_now_out

		return bg_prev, bg_now