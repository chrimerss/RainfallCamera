import os
import cv2
from model import RainNet
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.utils as utils
import torch.nn as nn
import numpy as np
from utils import print_network, init_weights, Dataset, RainLoss
from torch.utils.data import DataLoader


def dataprep(batch_size):
	folder_pt= 'datasets/'
	folders= os.listdir(folder_pt)

	for folder in folders:
		print(folder+ '...')
		src= os.path.join(folder_pt, folder)
		images= os.listdir(src)
		ind=0
		while True:
			try:
				imgs_prev= []
				imgs_now= []
				for b in range(batch_size):
					img_prev= cv2.imread(os.path.join(src,images[ind+b]))
					img_now= cv2.imread(os.path.join(src,images[ind+1+b]))
					img_prev= cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
					img_now= cv2.cvtColor(img_now, cv2.COLOR_BGR2GRAY)
					imgs_prev.append(img_prev/255.)
					imgs_now.append(img_now/255.)
				
				ind+=batch_size
				imgs_prev= np.array(imgs_prev)[:,np.newaxis,:,:]
				imgs_now= np.array(imgs_now)[:,np.newaxis,:,:]

				yield torch.Tensor(imgs_prev), torch.Tensor(imgs_now)

			except IndexError:
				break

def train(use_gpu):
	#load data
	bsize=1
	model= RainNet(bsize=bsize, use_gpu=True)
	model_path= 'logs/RainNet_loss1.pth'
	num_epochs= 100
	print_network(model)
	#initialize weights
	model.apply(init_weights)

	data= Dataset()
	print("training samples:", len(data))
	loader_train= DataLoader(dataset= data, batch_size=bsize, shuffle=True)

	criterion= RainLoss()
	if use_gpu:
		model= model.cuda()
		criterion.cuda()

	optimizer= optim.Adam(model.parameters(), lr=1e-4)
	scheduler= MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.1)

	writer= SummaryWriter('logs/RainLoss')
	steps=0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-'*20)
		for params in optimizer.param_groups:
			print('Learning rate: %f' %params['lr'])


		for i, (imgs_prev, imgs_now) in enumerate(loader_train):
			# input size: (4,1,401,401)
			assert imgs_prev.shape==(bsize,1,401,401),\
						 'wrong input size, expected (%d,1,401,401), but get %s'%(bsize, imgs_prev.shape)
			imgs_prev, imgs_now= Variable(imgs_prev), Variable(imgs_now)
			if use_gpu:
				imgs_prev, imgs_now= imgs_prev.cuda(), imgs_now.cuda()
			# print('before transform: ',imgs_prev.size())
			imgs_prev, imgs_now= torch.unsqueeze(imgs_prev, dim=1), torch.unsqueeze(imgs_now, dim=1)
			# print('transformed:', imgs_prev.size())
			inputs= torch.cat([imgs_prev, imgs_now], dim=1) #(bsize,2,1,401,401)
			# print('after stack:',inputs.size())
			# assert np.array(inputs.cpu().detach()).shape==(bsize,2,1,401,401),\
			# 			'expected input size (%d,2,1, 401,401) but got %s'%(bsize, str(inputs.size()))
			# test training samples:
			# cv2.imshow('prev',np.array(imgs_prev.cpu().detach().squeeze()*255.).astype(np.uint8))
			# cv2.imshow('now',np.array(imgs_now.cpu().detach().squeeze()*255.).astype(np.uint8))
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			optimizer.zero_grad()
			model.train()

			bg_prev, bg_now,rain_prev, rain_now= model(inputs)
			bg_prev, bg_now, rain_prev, rain_now= bg_prev*255., bg_now*255.,rain_prev*255.,rain_now*255.
			loss= criterion(rain_prev, bg_prev, rain_now, bg_now,)

			loss.backward()
			optimizer.step()

			model.eval()
			bg_prev, bg_now,rain_prev, rain_now= model(inputs)
			rain_prev,rain_now= torch.clamp(rain_prev,0.,1.),torch.clamp(rain_now,0.,1.)
			print("[epoch %d/%d][%d/%d]        loss: %.4f"%(epoch+1,num_epochs, i+1, len(data)//bsize, loss.item()))

			if steps%5==0:
				writer.add_scalar('loss', loss.item(), steps)
			steps+=1

		model.eval()
		bg_prev, bg_now,rain_prev, rain_now= model(inputs)
		rain_prev,rain_now= torch.clamp(rain_prev,0.,1.),torch.clamp(rain_now,0.,1.)
		rainy= utils.make_grid(inputs.data[-1,1,0,:,:], nrow=8, normalize=True, scale_each=True)
		rain_streak= utils.make_grid(rain_now.data.squeeze(),nrow=8, normalize=True, scale_each=True)
		bg= utils.make_grid(bg_now.data.squeeze(),nrow=8, normalize=True, scale_each=True)
		writer.add_image('rain', rain_streak, epoch+1)
		writer.add_image('origin', rainy, epoch+1)
		writer.add_image('background',bg,epoch+1)


	torch.save(model.state_dict(),model_path)


if __name__=='__main__':
	train(True)
