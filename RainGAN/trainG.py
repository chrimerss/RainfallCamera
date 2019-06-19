#! usr/bin/python
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.utils as utils
from torch.utils.data import DataLoader
from model import RainNet
from dataprep import Data_train_Gen
import cv2
import numpy as np
from loss import SSIM


def train():

	def init_weights(m):
		classname= m.__class__.__name__

		if classname.find('Conv')!=-1:
			m.weight.data.fill_(0.0)
			m.bias.data.fill_(0.0)

	bsize=2
	epoches=100
	use_gpu=True
	#prepare data
	data= Data_train_Gen()
	loader_train= DataLoader(dataset= data, batch_size=bsize, shuffle=True)
	# print(data.parameters())
	print('training samples: ', len(data))

	#model
	model= RainNet(use_gpu)
	model.apply(init_weights)

	criterion= SSIM()

	if use_gpu:
		model= model.cuda()
		criterion= criterion.cuda()
	#optimizer

	# print(list(model.parameters()))
	optimizer= torch.optim.Adam(model.parameters(), lr = 1e-3)
	scheduler= MultiStepLR(optimizer, milestones=[60], gamma=0.1)

	#tensorboard setup
	writer= SummaryWriter('logs/RainNet_G')

	for epoch in range(epoches):
		print("Epoch %d/%d"%(epoch, epoches) )
		print('-'*20)
		
		for params in optimizer.param_groups:
			print('learning rate: ', params['lr'])

		for i, (input, target) in enumerate(loader_train):
			a= list(model.parameters())[2].clone()
			# assert train.size()==(bsize,1,401,401),'invalid training data shape %s'%str(train.size())
			optimizer.zero_grad()
			model.train()
			input, target= Variable(input), Variable(target)
			if use_gpu:
				input, target= input.cuda(), target.cuda()
			# print(input.sum(),target.sum())
			bg_prev, bg_now, _, _= model(input)
			# print(bg_prev.sum(), target.sum(), rain_prev.sum())

			# bg_prev, bg_now= (bg_prev*255.).type(torch.uint8), (bg_now*255.).type(torch.uint8)

			# target= (target*255.).type(torch.uint8)
			# cv2.imshow('bg', bg_prev[0,:,:,:].detach().cpu().numpy().squeeze().astype(np.uint8))
			# cv2.imshow('truth',(target[0,0,:,:,:].detach().cpu().numpy().squeeze()).astype(np.uint8))
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			# target= target.squeeze()
			# print(out.size(), target.size())

			loss= -criterion(bg_prev, bg_now, target)

			loss.backward()
			optimizer.step()

			print("[%d/%d][%d/%d]            loss:%.4f/%.1f"%(epoch,epoches,i,len(data)//bsize, loss.item(),bsize*1*2))
			if i%5==0:
				writer.add_scalar('loss', loss.item(), i+1)

			b=list(model.parameters())[2].clone()
			# print(a.data.sum(), b.data.sum())
			assert torch.equal(a.data, b.data),'parameters are not updated! '

		# if epoch%50==0:
		# 	# print(list(model.parameters())[0].grad)
		# 	# print(model.clsnet[3].weight.grad)
		# 	batch_num= np.random.randint(0,bsize)
		# 	model.eval()
		# 	label= model(train)[batch_num].squeeze()
		# 	if label>0.5:
		# 		text= 'rain'
		# 	else:
		# 		text='no rain'
		# 	# print(target[batch_num,:],label, text)
		# 	img= utils.make_grid(train.data[batch_num,:,:,:], nrow=8, normalize=True, scale_each=True)

			# writer.add_image(text,img, epoch+1)
		scheduler.step()
		for name, param in model.named_parameters():
			writer.add_histogram(name, param.clone().cpu().data.numpy(), i+1)

		model.eval()
		bg_prev, bg_now, rain_prev, _= model(input)
		gt_prev, gt_now= target[0,0,:,:,:].clone(), target[0,1,:,:,:].clone()
		bg_prev, bg_now= torch.clamp(bg_prev, 0.,1.),torch.clamp(bg_now, 0.,1.)
		rain_prev= torch.clamp(rain_prev,0.,1.)
		label= utils.make_grid(bg_prev, nrow=8, normalize=True, scale_each=True)
		gt= utils.make_grid(gt_prev, nrow=8, normalize=True, scale_each=True)
		rain= utils.make_grid(rain_prev, nrow=8, normalize=True,scale_each=True)

		writer.add_image('cleaned',label, epoch+1)
		writer.add_image('ground_truth', gt, epoch+1)
		writer.add_image('rain',rain,epoch+1)

		b= list(model.parameters())[3].clone()
		assert not torch.equal(a.data, b.data),'parameters are not updated! '

	torch.save(model.state_dict(), 'logs/Generator-pretrain.pth')



if __name__=='__main__':
	train()

