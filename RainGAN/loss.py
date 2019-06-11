import torch
import torch.nn as nn
import torch.nn.functional as F

class RainLoss(nn.Module):
	def __init__(self, bsize, use_gpu=True):
		super(RainLoss,self).__init__()
		self.bsize=bsize
		self.use_gpu= use_gpu


	def forward(self, bg_prev, bg_now, rain_now, D_rain, D_bg):
		# term 1: background consistency
		term_1= F.l1_loss(bg_prev, bg_now)
		term_2= torch.zeros(1)
		term_3= torch.zeros(1)
		lambda1= 0.4
		lambda2= 0.3
		lambda3= 0.3
		if self.use_gpu:
			term_2=  term_2.cuda()
			term_3= term_3.cuda()
		for b in range(self.bsize):
			#term 2: rain streak GAN loss
			term_2+= 1.0-D_rain[b]
		#term 3: bg GAN loss
			term_3+= D_bg[b]


		print('Generator loss: %.4f, rain discriminator loss: %.4f, background discriminator loss %.4f'%(
			term_1,term_2,term_3))


		return lambda1*term_1+ lambda2*term_2+ lambda3*term_3