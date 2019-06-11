#! usr/bin/python
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.utils as utils
from torch.utils.data import DataLoader
from model import RainDescriminator
from dataprep import DataSet
import cv2
import numpy as np


def train():

	def init_weights(m):
		classname= m.__class__.__name__

		if classname.find('Conv')!=-1:
			m.weight.data.uniform_(0.0,1.0)
			m.bias.data.fill_(0)

	bsize=6   #total training samples 522
	epoches=200
	use_gpu=True
	show_img=False
	#prepare data
	data= DataSet(datatype='rain')
	loader_train= DataLoader(dataset= data, batch_size=bsize, shuffle=True)
	# print(data.parameters())
	print('training samples: ', len(data))

	#model
	model= RainDescriminator(use_gpu)
	model.apply(init_weights)

	criterion= nn.BCELoss()

	if use_gpu:
		model= model.cuda()
		criterion= criterion.cuda()
	#optimizer

	# print(list(model.parameters()))
	optimizer= torch.optim.Adam(model.parameters(), lr = 1e-3)
	scheduler= MultiStepLR(optimizer, milestones=[100], gamma=0.1)

	#tensorboard setup
	writer= SummaryWriter('logs/BackGroundDiscriminator-rainstreak')

	for epoch in range(epoches):
		print("Epoch %d/%d"%(epoch, epoches) )
		print('-'*20)
		a= list(model.parameters())[0].clone()
		for params in optimizer.param_groups:
			print('learning rate: ', params['lr'])

		for i, (train, target) in enumerate(loader_train):

			assert train.size()==(bsize,1,401,401),'invalid training data shape %s'%str(train.size())
			optimizer.zero_grad()
			model.train()
			if use_gpu:
				train, target= train.cuda(), target.cuda()
			out= model(train)
			# target= target.squeeze()
			# print(out.size(), target.size())
			# print(out, target)
			loss=  criterion(out, target)

			loss.backward()
			optimizer.step()
			

			print("[%d/%d][%d/%d]            loss:%.4f"%(epoch,epoches,i,len(data)//bsize, loss.item()))
			if i%5==0:
				writer.add_scalar('loss', loss.item(), i+1)

		#update scheduler
		scheduler.step()

		if epoch%50==0:
			# print(list(model.parameters())[0].grad)
			# print(model.clsnet[3].weight.grad)
			
			batch_num= np.random.randint(0,bsize)
			model.eval()
			label= model(train)[batch_num].squeeze()
			if label>0.5:
				text= 'rain'
			else:
				text='no rain'
			# print(target[batch_num,:],label, text)
			img= utils.make_grid(train.data[batch_num,:,:,:], nrow=8, normalize=True, scale_each=True)

			writer.add_image(text,img, epoch+1)

		b= list(model.parameters())[0].clone()
		assert not torch.equal(a.data, b.data),'parameters are not updated! '

	torch.save(model.state_dict(), 'logs/RainDiscriminator.pth')



if __name__=='__main__':
	train()




