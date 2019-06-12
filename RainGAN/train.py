import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.utils as utils
from torch.utils.data import DataLoader
from model import BGDescriminator,RainDescriminator,RainNet   #load three models
from dataprep import DataSet_GAN
from loss import RainLoss, RainLoss2
import cv2
import numpy as np
import argparse

CONFIG= {
	'use_gpu':True,
	'batch_size':2,
	'epoches':500,
	'model_path_gan':'logs/generator-6-12.pth',
	'bg_D_path':'logs/BGDiscriminator.pth',
	'rain_D_path':'logs/RainDiscriminator.pth',
	'lr':1e-3,
	'scheduled_milestone':list((100, 200, 400)),
	'writer': True,
	'gamma':0.1,
	'writer_path':'logs/RainGAN-loss_ssim-gan',
	'save_progressive_model': True
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

	# load data
	data= DataSet_GAN()
	train_dataloader= DataLoader(dataset= data, batch_size=bsize, shuffle=True)
	#load models
	Generator= RainNet(bsize, use_gpu)
	bg_D= BGDescriminator( use_gpu)
	rain_D= RainDescriminator(use_gpu)
	print('total samples :' , len(data))
	# print('model structure: ', Generator)
	num_params(Generator)


	Generator.apply(init_weights) #initialize model

	#loss function
	criterion= RainLoss2(bsize)

	if use_gpu:
		Generator= Generator.cuda()
		bg_D= bg_D.cuda()
		rain_D= rain_D.cuda()
		criterion= criterion.cuda()
	
	bg_D.load_state_dict(torch.load(cmd.bg_D_path))
	rain_D.load_state_dict(torch.load(cmd.rain_D_path))

	#optimizer
	optimizer= torch.optim.Adam(Generator.parameters(), lr=1e-3)
	scheduler= MultiStepLR(optimizer, milestones=cmd.scheduled_milestone, gamma=cmd.gamma)

	#tensorboard set-up
	if cmd.writer:
		writer= SummaryWriter(cmd.writer_path)
	steps=0

	for epoch in range(epoches):

		print('-'*30)
		for param in optimizer.param_groups:
			print('learning rate: ', param['lr'])

		for i,(imgs_prev, imgs_now) in enumerate(train_dataloader):

			optimizer.zero_grad()
			Generator.train()
			bg_D.eval()
			rain_D.eval()

			imgs_prev, imgs_now= Variable(imgs_prev), Variable(imgs_now)

			if use_gpu:
				imgs_prev, imgs_now= imgs_prev.cuda(), imgs_now.cuda()
			# print('before transform: ',imgs_prev.size())
			imgs_prev, imgs_now= torch.unsqueeze(imgs_prev, dim=1), torch.unsqueeze(imgs_now, dim=1)
			# print('transformed:', imgs_prev.size())
			inputs= torch.cat([imgs_prev, imgs_now], dim=1) #(bsize,2,1,401,401)

			bg_prev, bg_now,rain_prev, rain_now= Generator(inputs)

			bg_prev= bg_prev*255.
			bg_now= bg_now*255.

			#mask rain streak
			rain_now= mask(rain_now)

			D_bg= bg_D(bg_now)
			D_rain= rain_D(rain_now)

			loss= criterion(imgs_now, bg_prev, bg_now, rain_now, D_rain, D_bg)

			loss.backward()
			optimizer.step()

			print(
				'[%d/%d][%d/%d]    Total loss: %.4f    Accuracy: %.2f '%(
				epoch, epoches, i, len(data)//bsize, loss.item(), -loss.item()
																		)
				)

			if i%10==0 and cmd.writer:
				writer.add_scalar('loss',loss.item(),steps+1)

			steps+=1
			del bg_now, bg_prev, rain_prev, rain_now

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

		scheduler.step()

		if epoch%100==0 and cmd.save_progressive_model:
			torch.save(Generator.state_dict(), 'generator-epoch-%d-6-10.pth'%epoch)

	torch.save(Generator.state_dict(), model_path)




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