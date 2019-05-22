import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import sys
sys.path.append('/PReNet')
from PReNet.dataprep import *
from PReNet.utils import *
from torch.optim.lr_scheduler import MultiStepLR
from PReNet.SSIM import SSIM
from PReNet.generator import *

if torch.cuda.is_available():
	print('using gpu training ...')

def main():
	print('Loading dataset ...\n')
	dataset_train= Dataset(data_path= 'datasets/')
	loader_train= DataLoader(dataset= dataset_train)
	print('# of training samples :', int(len(loader_train)))
	# define some hyper-parameters
	recurr_iter= 4
	use_GPU= True
	model_path= 'logs/real/PReNet1.pth'
	num_epochs= 5

	model= Generator_lstm(recurr_iter, use_GPU)
	print_network(model)
	model.load_state_dict(torch.load(model_path))

	#loss
	criterion= SSIM()

	if use_GPU:
		model= model.cuda()
		criterion.cuda()

	#optimizer:
	optimizer= optim.Adam(model.parameters(), lr= 1e-4)
	scheduler = MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)

	#record training
	writer= SummaryWriter('logs/')
	step=0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for i, (input_train, target_train) in enumerate(loader_train,0):
			input_train, target_train= Variable(input_train), Variable(target_train)
			if use_GPU:
				input_train, target_train= input_train.cuda(), target_train.cuda()
			optimizer.zero_grad()

			model.train()
			out_train,_ =model(input_train)
			pixel_metric= criterion(target_train, out_train)
			loss= -pixel_metric

			loss.backward()
			optimizer.step()

			model.eval()
			out_train, _ = model(input_train)
			out_train = torch.clamp(out_train, 0., 1.)
			print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item()))

			if step%10==0:
				writer.add_scalar('loss', loss.item(), step)
			step+=1

		model.eval()
		out_train,_ = model(input_train)
		out_train= torch.clamp(out_train,0.,1.)
		im_target= utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
		im_input= utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
		im_derain= utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
		writer.add_image('clean image', im_target, epoch+1)
		writer.add_image('rainy image', im_input, epoch+1)
		writer.add_image('derained image', im_derain, epoch+1)

		torch.save(model.state_dict(), 'logs/real/latest.pth')


if __name__ == '__main__':
	main()