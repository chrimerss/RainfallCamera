import torch.nn as nn
import torch



class Generator(nn.Module):
	def __init__(self,input_vec=4,use_gpu=True):
		super(Generator, self).__init__()
		self.input_vec= input_vec

		self.fc= nn.Sequential(
								nn.Linear()
									)

	def forward(self, cat, img):

		bsize, channel, m,n= img.size()
		#cat shape (4,) one-hot
		#img data shape (bsize, channel, m, n)
		


		pass


class Descrimitor(nn.Module):
	def __init__(self, use_gpu=True):
		pass

	def forward(self,x):
		#input data shape (bsize, channel, m, n)
		pass