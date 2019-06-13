"""
This module makes discriminator trainable

version: 0.1
"""

'''
This module trains the main RainGAN.

version: 0.0

'''

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.utils as utils
from torch.utils.data import DataLoader
from model import BGDescriminator,RainDescriminator,RainNet   #load three models
from dataprep import GanData
from loss import RainLoss, GenLoss2, DisLoss
import cv2
import numpy as np
import argparse

CONFIG= {
	'use_gpu':True,
	'batch_size':2,
	'epoches':500,
	'model_path_gan':'logs/generator-6-13.pth',
	'model_path_D_rain': 'logs/discriminator-rain-6-13',
	'model_path_D_bg': 'logs/discriminator-bg-6-13',
	'bg_D_path':'logs/BGDiscriminator.pth',
	'rain_D_path':'logs/RainDiscriminator.pth',
	'lr':1e-3,
	'scheduled_milestone':list((100, 200, 400)),
	'writer': True,
	'gamma':0.1,
	'writer_path':'logs/RainGAN-loss_ssim-dis_train',
	'save_progressive_model': True,
	'chkpnts_milestone': 100
	}

def cmd_args():
	global CONFIG
	parser= argparse.ArgumentParser('training setting')
	parser.add_argument('--use_gpu', default=CONFIG['use_gpu'],type=bool, help='whether use gpu')
	parser.add_argument('--batch_size', default=CONFIG['batch_size'],type=int, help='batch_size')
	parser.add_argument('--epoches', default=CONFIG['epoches'], type=int, help='# of epoches')
	parser.add_argument('--model_path_gan', default=CONFIG['model_path_gan'],type=str, help='where to store model')
	parser.add_argument('--bg_D_path', default=CONFIG['bg_D_path'], type=str, help='where the background discriminator model stored')
	parser.add_argument('--rain_D_path', default=CONFIG['rain_D_path'], type=str, help='where the rain discriminator model stored')
	parser.add_argument('--lr', default=CONFIG['lr'], type=float,help='learning rate')
	parser.add_argument('--scheduled_milestone', type=str, default=CONFIG['scheduled_milestone'], help='list of milestones')
	parser.add_argument('--gamma', default=CONFIG['gamma'], type=float,help='gamma for scheduled lr')	
	parser.add_argument('--writer', default=CONFIG['writer'], type=bool, help='tensorboard writer')
	parser.add_argument('--writer_path', default=CONFIG['writer_path'], help='where to put writer')
	parser.add_argument('--save_progressive_model', default=CONFIG['save_progressive_model'], help='determine to save progressive results')
	parser.add_argument('--chkpnts_milestone', default=CONFIG['chkpnts_milestone'],type=int,help='milestones to store checkpoints')
	parser.add_argument('--model_path_D_rain', default=CONFIG['model_path_D_rain'],type=str,help='store rain discriminator model path')
	parser.add_argument('--model_path_D_bg', default=CONFIG['model_path_D_bg'],type=str,help='store bg discriminator model path')

	return parser.parse_args()


def train():
	#hyperparameters
	cmd= cmd_args()
	print('|-------------------Model Training Setup--------------|\n|use GPU:  %r                                       |\n\
|batch size:  %d                                       |\n|epoches:  %d                                        |\n\
|learning rate:  %.6f                             |\n|gamma:    %.2f                                       |\n\
|-----------------------------------------------------|'%(cmd.use_gpu,cmd.batch_size,cmd.epoches,cmd.lr,cmd.gamma))

	use_gpu=cmd.use_gpu
	bsize=cmd.batch_size
	epoches= cmd.epoches
	model_path=cmd.model_path_gan
	model_path_D_bg= cmd.model_path_D_rain
	model_path_D_rain= cmd.model_path_D_bg

	# load data
	data= GanData()
	train_dataloader= DataLoader(dataset= data, batch_size=bsize, shuffle=True)
	#load models
	Generator= RainNet(bsize, use_gpu)
	bg_D= BGDescriminator(use_gpu)
	rain_D= RainDescriminator(use_gpu)
	# print('total samples :' , len(data))
	# print('model structure: ', Generator)
	print('generator:')
	num_params(Generator)
	print('background discriminator:')
	num_params(bg_D)
	print("rainstreak discriminator")
	num_params(rain_D)


	Generator.apply(init_weights) #initialize model

	#loss function
	g_criterion= GenLoss2(bsize)
	d_criterion= DisLoss(bsize)
	# d_criterion= 

	if use_gpu:
		Generator= Generator.cuda()
		bg_D= bg_D.cuda()
		rain_D= rain_D.cuda()
		g_criterion= g_criterion.cuda()
		d_criterion= d_criterion.cuda()
	
	bg_D.load_state_dict(torch.load(cmd.bg_D_path))
	rain_D.load_state_dict(torch.load(cmd.rain_D_path))

	#optimizer
	optimizer_G= torch.optim.Adam(Generator.parameters(), lr=1e-3)
	optimizer_D= torch.optim.Adam([
								{'params':bg_D.parameters(), 'lr':1e-4},
								{'params':rain_D.parameters(), 'lr':1e-4},
								])
	scheduler_G= MultiStepLR(optimizer_G, milestones=cmd.scheduled_milestone, gamma=cmd.gamma)
	scheduler_D= MultiStepLR(optimizer_D, milestones=cmd.scheduled_milestone, gamma=cmd.gamma)

	#tensorboard set-up
	if cmd.writer:
		writer= SummaryWriter(cmd.writer_path)
	steps=0

	for epoch in range(epoches):

		print('-'*30)
		for param in optimizer_G.param_groups:
			print('generator learning rate: ', param['lr'])
		
		for param in optimizer_D.param_groups:
			print('discriminator learning rate: ', param['lr'])

		for i,((imgs_prev, imgs_now),(real_rain, label_rain),(real_bg, label_bg)) in enumerate(train_dataloader):

			optimizer_D.zero_grad()
			
			bg_D.train()
			rain_D.train()

			imgs_prev, imgs_now= Variable(imgs_prev), Variable(imgs_now)

			if use_gpu:
				imgs_prev, imgs_now= imgs_prev.cuda(), imgs_now.cuda()
				real_rain, label_rain= real_rain.cuda(), label_rain.cuda()
				real_bg, label_bg= real_bg.cuda(), label_bg.cuda()
			# print('before transform: ',imgs_prev.size())
			imgs_prev, imgs_now= torch.unsqueeze(imgs_prev, dim=1), torch.unsqueeze(imgs_now, dim=1)
			# print('transformed:', imgs_prev.size())
			inputs= torch.cat([imgs_prev, imgs_now], dim=1) #(bsize,2,1,401,401)

			bg_prev, bg_now,rain_prev, rain_now= Generator(inputs)

			prev_bg= bg_prev.clone()*255.
			now_bg= bg_now.clone()*255.

			# #mask rain streak
			# rain_now= mask(rain_now)

			D_bg= bg_D(bg_now)
			real_bg_D= bg_D(real_bg)
			D_rain= rain_D(rain_now)
			real_rain_D= rain_D(real_rain)

			d_loss= d_criterion(real_rain_D, label_rain, D_rain, real_bg_D, label_bg, D_bg)

			# print(dr_loss, db_loss)
			d_loss.backward(retain_graph=True)

			optimizer_D.step()

			Generator.train()

			optimizer_G.zero_grad()

			g_loss= g_criterion(imgs_now, prev_bg, now_bg, rain_now, D_rain, D_bg)

			g_loss.backward()
			optimizer_G.step()

			print(
				'[%d/%d][%d/%d]    Generator loss: %.4f    discriminator loss: %.4f'%(
				epoch, epoches, i, len(data)//bsize, g_loss.item(), d_loss.item())
				)

			if i%10==0 and cmd.writer:
				writer.add_scalar('generator loss',g_loss.item(),steps+1)
				writer.add_scalar('discriminator loss',d_loss.item(),steps+1)


			steps+=1



		if cmd.writer:
			batch_num= np.random.randint(0,bsize)
			Generator.eval()
			bg_prev, bg_now,rain_prev, rain_now= Generator(inputs)
			rain_prev,rain_now= torch.clamp(rain_prev,0.,1.),torch.clamp(rain_now,0.,1.)
			rainy= utils.make_grid(inputs.data[batch_num,1,0,:,:], nrow=8, normalize=True, scale_each=True)
			if bsize>1:
				rain_streak= utils.make_grid(rain_now.data.squeeze()[batch_num,:,:],nrow=8, normalize=True, scale_each=True)
				bg= utils.make_grid(bg_now.data.squeeze()[batch_num,:,:],nrow=8, normalize=True, scale_each=True)
			elif bsize==1:
				rain_streak= utils.make_grid(rain_now.data.squeeze(),nrow=8, normalize=True, scale_each=True)
				bg= utils.make_grid(bg_now.data.squeeze(),nrow=8, normalize=True, scale_each=True)
			
			writer.add_image('rain', rain_streak,epoch+1)
			writer.add_image('origin', rainy, epoch+1)
			writer.add_image('background',bg,epoch+1)

		scheduler_G.step()
		scheduler_D.step()

		if epoch%50==0 and cmd.save_progressive_model:
			torch.save(Generator.state_dict(), 'generator-epoch-%d-6-10.pth'%epoch)

	torch.save(Generator.state_dict(), model_path)
	torch.save(D_bg.state_dict(), model_path_D_bg)
	torch.save(D_rain.state_dict(), model_path_D_rain)

	return None

def init_weights(m):
	classname= m.__class__.__name__

	if classname.find('conv2d')!=-1:
		m.weight.data.uniform_(0.0,1.0)
		m.bias.data.fill_(0.0)

def mask(tensor):
	mask= tensor.ge(0.2)
	tensor[mask]=1
	tensor[~mask]=0

	return tensor

def num_params(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

if __name__=='__main__':
	train()