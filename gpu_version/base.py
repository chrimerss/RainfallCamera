import torch
import numpy as np
import cv2

class Base(object):
	'''
	Implementation of some useful functions used for tensor calculation,
	assertion.
	'''

	def assert_tensor(self, *tensors):
		for tensor in tensors:
			if not isinstance(tensor, torch.Tensor):
				raise ValueError(f'provided input is not tensor object, received {type(tensor)}')

	def assert_color_img(self, *imgs):
		for img in imgs:
			if len(img.shape)!=3 or img.shape[-1]!=3:
				raise ValueError(f'Expected colored image, but get image shape {img.shape}')


	def assert_gray_img(self, *img):
		for img in imgs:
			if len(img.shape)!=2:
				raise ValueError(f'Expected gray scale image, but get image shape {img.shape}')

	def assert_UMat(self, *srcs):
		for src in srcs:
			if not isinstance(src, cv2.UMat):
				raise ValueError(f"Expected cv2.UMat object, but get type {type(src)}")

	def assert_ndarray(self, *srcs):
		for src in srcs:
			if not isinstance(src, np.ndarray):
				raise ValueError(f'Expected numpy.ndarray object but get type {type(src)}')

	def assert_int(self, *args):
		for arg in args:
			if not isinstance(arg, int):
				raise ValueError(f'Expected int object but get type {type(arg)}')

	def assert_float(self, *args):
		for arg in args:
			if not isinstance(arg, float):
				raise ValueError(f'Expected float object but get type {type(arg)}')

	def numpy2tensor(self, src):
		self.assert_ndarray(src)

		return torch.from_numpy(src).cuda()

	def umat2numpy(self, src):
		self.assert_UMat(src)

		return cv2.UMat.get(src)

	def umat2tensor(self, src):
		arr= self.umat2numpy(src).astype(np.int64)
		tensor= self.numpy2tensor(arr).cuda()

		return tensor

	def tensor2numpy(self,src):
		self.assert_tensor(src)

		return src.numpy()

	def numpy2umat(self, src):
		self.assert_ndarray(src)

		return cv2.UMat(src)

	def tensor2umat(self, src):
		arr= self.tensor2numpy(src)
		umat= self.numpy2umat(arr)

		return umat

	def cvt_BGR2GRAY(self, src):
		self.assert_UMat(src)

		return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

