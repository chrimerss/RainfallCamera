import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

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

class GenLoss2(nn.Module):
	'''SSIM + Rain Discriminator + Background Discriminator'''
	def __init__(self, bsize, use_gpu=True, window_size = 11, size_average = True):
		super(GenLoss2, self).__init__()
		self.bsize= bsize
		self.use_gpu=use_gpu
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, imgs_now, bg_prev, bg_now, rain_now, D_rain, D_bg):
		# term 1 
		window= self.window.cuda() if self.use_gpu else self.window
		# ssim= msssim(bg_now, bg_prev, self.window_size, self.size_average, normalize=True)
		now_img= imgs_now[:,0,:,:,:].clone()*255.
		ssim= _ssim(now_img, bg_now, window, self.window_size, self.channel, self.size_average)
		term_1= F.l1_loss(bg_prev, bg_now)
		#term 2
		term_2= torch.zeros(1)
		term_3= torch.zeros(1)
		lambda1= 0.4
		lambda2= 0.2
		lambda3= 0.3
		lambda4= 0.3
		if self.use_gpu:
			term_2=  term_2.cuda()
			term_3= term_3.cuda()
		for b in range(self.bsize):
			#term 2: rain streak GAN loss
			term_2+= 1.0-D_rain[b]
		#term 3: bg GAN loss
			term_3+= D_bg[b]

		print('Generator ssim loss: %.4f, Generator l1 loss: %.4f rain discriminator loss: %.4f, background discriminator loss %.4f'%(
			-ssim,term_1,term_2,term_3))

		return lambda1*(-ssim)+ lambda2*term_1+ lambda3*term_2+ lambda4*term_3

class DisLoss(nn.Module):
    def __init__(self):
        super(DisLoss, self).__init__()

    def forward(self, ):
        pass


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2
    # print((2*mu1_mu2 + C1)*(2*sigma12 + C2))
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous(), requires_grad=True)

    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret