'''
GPU version of implementation on PCA with single image
'''

import torch
import numpy as np
import cv2
from base import Base

torch.cuda.init()

if torch.cuda.is_available():
	print('using GPU ...')

class PCA_GPU(Base):

	def __init__(self, img_path=None):
		if img_path:
			self.img= cv2.imread(img_path).squeeze()
			self.img= cv2.UMat(self.img)
			self.img= self.cvt_BGR2GRAY(self.img)
			

	def _connected_components(self, img):
		self.assert_UMat(img)

		return cv2.connectedComponents(img)

	def _location_vec(self, ret, labels):
		'''
		---------------
		Args:
			labels, X, Y: torch.tensor object

		'''

		h, w= labels.shape
		first= True
		for label in torch.unique(labels):
			if first:
				X_series, Y_series= self._where(labels, label)
				first= False
			else:
				_X, _Y= self._where(labels, label)
				X_series= torch.cat([X_series, _X])
				Y_series= torch.cat([Y_series, _Y])

		return X_series, Y_series


	def _covariance(self, labels, X, Y, c):
		'''
		---------------
		Args:
			labels, X, Y: torch.tensor object
		'''
		self.assert_tensor(labels, X, Y)
		indexs= []
		for i, label in enumerate(torch.unique(labels)):
			X_series, Y_series= self._where(labels, label)
			mat= torch.zeros((2,2)).cuda()
			for x, y in zip(X_series, Y_series):
				mat= torch.add(mat, (torch.Tensor([[x],[y]]).cuda()@torch.Tensor([[x,y]]).cuda()	)		
								)
			mat= torch.div(mat,len(X_series))
			comat= mat- (torch.Tensor([[X[i]],[Y[i]]]).cuda()@torch.Tensor([[X[i],Y[i]]]).cuda())
			l,w,theta= self._eigen_chara(comat, c)
			if self._refinement(l,w,theta):
				indexs.append(label)
		indexs= torch.Tensor(indexs)

		return indexs

	def _eigen_chara(self, mat, c):
		'''
		-------------
		Args:
			covariances: tensor; obtained by _covariance()
			c: int

		'''

		self.assert_tensor(mat)
		self.assert_float(c)
		assert mat.shape==(2,2), f'expected matrix shape (2,2) but received {mat.shape}'
		w,v= mat.eig(eigenvectors=True)
		theta= torch.atan(v[0][1]/v[0][0])
		l= c*w.max()
		w= c*w.min()

		return l,w,theta

	def _refinement(self, l, w, theta, max_width=0.5, l_w_ratio=10., abs_angle=45.):
		'''
		------------
		Args:
			L, W, theta: torch.tensor
			max_width, l_w_ratio, abs_angle: int
		'''
		self.assert_float(max_width, l_w_ratio, abs_angle)
		self.assert_tensor(l, w, theta)
		condition= (w<max_width) &(w>0) & (l>0) & (l/w>l_w_ratio) \
    						& (abs(theta)<abs_angle)
		
		if condition:
			return condition

	def _relocate(self, indexs, labels):
		self.assert_tensor(labels, indexs)
		first= True

		for label in indexs:
			_X, _Y= self._where(labels, label)
			print(_X, _Y)
			if first:
				X_ori, Y_ori= _X, _Y
				first= False
			else:
				X_ori, Y_ori= torch.cat([X_ori, _X]), torch.cat([Y_ori, _Y])
		print(X_ori, Y_ori)
		self.assert_tensor(X_ori, Y_ori)

		return X_ori, Y_ori 

	def execute(self):
		self.assert_UMat(self.img)
		print('connetcted components ...')
		ret, labels= self._connected_components(self.img)
		tensor= self.umat2tensor(labels)
		print('locating ...')
		X, Y= self._location_vec(ret, tensor)
		print('calculating covariance ...')
		indexs= self._covariance(tensor, X, Y, 0.5)
		print(indexs.shape)
		print('relocate ...')
		orig_X, orig_Y= self._relocate(indexs, tensor)
		img_new= torch.zeros(self.img.shape, dtype=torch.uint8).cuda()
		img_new[orig_X, orig_Y]= 255
		self.assert_tensor(img_new)

		return img_new

	def _where(self, tensor, target):
		'''
		mimic the numpy version where
		---------------
		Args:
			tensor: torch.Tensor obj
			target: torch.Tensor obj
		
		'''
		mask= tensor.ge(target)
		masked= torch.nonzero(mask)

		return masked[:,0], masked[:,1]


	# def _to_gray(self, src):
	# 	return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
	# def _UMat2Tensor(self, src):
	# 	self._assert_UMat(src)
	# 	arr= cv2.UMat.get(src).astype(np.int32)
	# 	tensor= torch.from_numpy(arr)

	# 	return tensor

	# def _assert_tensor(self, *srcs):
	# 	for src in srcs:
	# 		if not isinstance(src, torch.Tensor):
	# 			raise ValueError(f'Expected type torch tensor, but get {type(src)}')

	# def _assert_UMat(self, *srcs):
	# 	#assert image is black&white image
	# 	for src in srcs:
	# 		if not isinstance(src, cv2.UMat):
	# 			raise ValueError('Unknown provided input type. expected cv2.Mat')


	# def _assert_int(self, *nums):
	# 	for num in nums:
	# 		if not isinstance(num, int):
	# 			raise ValueError(f'object is supposed to be int but {type(num)} received')

	# def _assert_float(self, *nums):
	# 	for num in nums:
	# 		if not isinstance(num, float):
	# 			raise ValueError(f'object is supposed to be float but {type(num)} received')

	# def _assert_img(self, *imgs):
	# 	for img in imgs:
	# 		assert len(img.shape)==2, f'Please make sure provided image is black&white. received length {len(img.shape)}'
	# 		assert img.max()<=255, 'Please make sure provided image is 8 bit.'

if __name__=='__main__':
	img_path= 'D:\\Radar Projects\\lizhi\\CCTV\\Test_imgs\\20180324-0307\\Rain-6.png'
	pca= PCA_GPU(img_path)
	img= pca.execute()
	cv2.imshow('output', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
