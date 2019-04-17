'''
This module builds classification algorithm to sperate out four categories: [no-rain, noraml-rain, heavy-rain, night]
In order to classify them, SVM are desired with hyper-plane to seperate them out.
As for the distinguishable information to support it, we split each image into (10px,10px) sub-domain,
and inside each domain, we estimated the contract, minimum brightness, sharpness, mean hue, and mean saturation.
Overall, we are expected to feed (classes, images_in_class, num_domains, class_of_information) to SVM
with one-hot coded label.
This class uses Dask array as sufficient tool to train in SVM which is able to process large amount of data.

'''
import cv2
import numpy as np
# import scikit
from glob import glob
import dask
import dask.array as da
import dask_image.imread
from itertools import product
from dask.distributed import Client,progress, LocalCluster
import dask_ml.model_selection as dcv
from sklearn.svm import SVC
import pickle
import argparse
from joblib import dump, load


__all__=['Classifier']
__version__= 0.0
__author__= 'lizhi'

parser = argparse.ArgumentParser(description="SVM model")
parser.add_argument('--window_size', type=tuple, default=(10,10),help='window size default (10,10)')
parser.add_argument('--workers',type=int, default=8,help='determine how many threads to work')
parser.add_argument('--client', type=bool, default=False, help='specify need to build client to monitor')
parser.add_argument('--verbose', type=bool, default=False, help='verbose when implementing SVM')
parser.add_argument('--grid_search', type=bool, default=False, help='tune hyperparameters with dask')
parser.add_argument('--train', type=bool, default=False, help='train the model if not trained before.')
parser.add_argument('--test', type=bool, default=False, help='test the model if trained before.')
parser.add_argument('--model', type=str, default='svm_model-1-scale.joblib', help='find the model stored previously')

OPT = parser.parse_args()

if OPT.test:
	if OPT.model is None:
		raise FileNotFoundError('please specify the model trained before testing model.')

class Classifier:
	'''
	So far, we only consider four classes to specify.
	There is no need to define arguments when construct the instance.
	'''
	def __init__(self, window_size, whether_client, workers,verbose, grid_search, model_path):
		self.whether_client= whether_client
		self.window_size= window_size
		self.verbose= verbose
		self.grid_search=grid_search
		self.model_path= model_path
		self.workers= workers

	def _initialize(self):
		self.no_rain_path= 'datasets/no_rain/*.png'
		self.normal_path= 'datasets/normal/*.png'
		self.heavy_path= 'datasets/heavy/*.png'
		self.night_path= 'datasets/night/*.png'
		self.datasets= self.cat_arrays()
		self.num_folders, self.num_imgs= self.datasets.shape[:2]
		self.tot_imgs= self.num_folders*self.num_imgs
		
		self.resize_shape=300 #reshape all images into (300,300)
		assert self.datasets.shape==(self.num_folders,self.num_imgs,1080,1920,3),f'get array shape {self.datasets.shape}, array may not be correct, please check.'
		# self.datasets= self.resize(self.datasets)
	def __repr__(self):
		return str(self.datasets)

	def train(self):
		self._initialize()
		print(self.tot_imgs,' images to process ...')
		# resize images:
		if self.whether_client:
			c= LocalCluster(n_workers=self.workers)
			print(c)
			self.client= Client(c)
			print(self.client)
			self.datasets= self.datasets.map_blocks(self.resize, chunks=(1,1,self.resize_shape,self.resize_shape,3), dtype=np.uint8)
		else:
			dask.config.set(scheduler='threads')
			self.datasets= self.datasets.map_blocks(self.resize, chunks=(1,1,self.resize_shape,self.resize_shape,3), dtype=np.uint8)
		tot_grids= (self.resize_shape//self.window_size[0])* (self.resize_shape//self.window_size[1])
		labels= [
				 ['no rain']*self.num_imgs,['normal']*self.num_imgs,
				 ['heavy']*self.num_imgs, ['night']*self.num_imgs
				]
		y= np.concatenate(labels)
		print('dask overview:  ',str(self.datasets))
		if self.whether_client:
			vectors= self.datasets.map_blocks(self.img_information, chunks=(1,5*tot_grids), drop_axis=[2,3,4], dtype=np.float32,
										 name='vectors').compute(scheduler='threads')
			vectors=self.client.persist(vectors)
			# print(vectors.gather())
			X= vectors.reshape((self.tot_imgs,5*tot_grids))
			params= {'C': [1,10],
						'kernel':['linear','rbf'],
						'degree':[3,5]}
			svc = SVC(verbose=self.verbose)
			clf = dcv.GridSearchCV(svc, params)
			clf.fit(X,y)
			pickle.dump(svm,'svm_model-1.pckl')

		else:
			vectors= self.datasets.map_blocks(self.img_information, chunks=(1,5*tot_grids), drop_axis=[2,3,4], dtype=np.float32,
										 name='vectors')
			# X= vectors.reshape((self.tot_imgs,5*tot_grids))
			X= vectors.compute(scheduler='threads')
			print('training samples dimension: ',X.shape,'\ntarget labels dimension',y.shape)
			# grid search for tuning hyper parameters
			if self.grid_search:
				params= {'C': [1,5]}
				svc = SVC(verbose=self.verbose)
				clf = dcv.GridSearchCV(svc, params)
				clf.fit(X,y)
				dump(clf, 'svm_model-0-grid_searched.joblib')
			else:
				svm= SVC(verbose=self.verbose, gamma='scale')
				svm.fit(X, y)
				dump(svm, 'svm_model-2-scale.joblib')


	def test(self, model, src):
		
		clf= load(model)
		src= cv2.resize(src, (300,300))
		test_img_info= self.img_information(src).squeeze().reshape(1,-1)
		result= clf.predict(test_img_info)

		return result

	def moving_window(self,src):
		m,n= self.window_size # m=10
		n_row= src.shape[0]//m  #30
		n_col= src.shape[1]//n  #30
		ind=-1
		for i in range(n_row):
			for j in range(n_col):
				sub_img= src[i*m:(i+1)*m, j*n:(j+1)*n,:]
				ind+=1

				yield sub_img, ind

	def img_information(self, block):
		# refer:https://en.wikipedia.org/wiki/Contrast_(vision)#Formula for contract
		#RMS contrast
		#brightness: minimum brightness stored
		#sharpness: represented by edges, we define it use the variance of the Sobel filter -> blurriness
		#returned value in a order ['contract','brightness','sharpness','hue','saturation']
		img= block.squeeze().copy()
		values= np.zeros((900,5), dtype=np.float32)
		for sub_img, ind in self.moving_window(img):
			ii,jj,mm= sub_img.shape
			for (h,w,c) in product(range(ii),range(jj),range(mm)):
				values[ind,0]= (
					((sub_img[h,w,c]-sub_img.mean())**2).sum()/sub_img.shape[0]/sub_img.shape[1]/sub_img.shape[2]
					)**0.5
				
			values[ind,1]= sub_img.min()
			values[ind,2]= cv2.Laplacian(sub_img,cv2.CV_64F).var()
			values[ind,3]= cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)[:,:,0].mean()
			values[ind,4]= cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)[:,:,1].mean()
		values= values.reshape(1,-1)
		return values[np.newaxis,:]

	def resize(self, block):
		img=block.squeeze().copy()

		return cv2.resize(img, (300,300))[np.newaxis, np.newaxis,:,:,:]

	def dask_array(self, pattern):
		return dask_image.imread.imread(pattern)

	def cat_arrays(self):
		no_rain= self.dask_array(self.no_rain_path)
		normal= self.dask_array(self.normal_path)
		night= self.dask_array(self.night_path)
		heavy= self.dask_array(self.heavy_path)
		datasets= da.stack([no_rain, normal, heavy, night], axis=0)
		return datasets

	def make_cluster(self,**kwargs):
	    try:
	        from dask_kubernetes import KubeCluster
	        kwargs.setdefault('n_workers', 8)
	        cluster = KubeCluster(**kwargs)
	    except ImportError:
	        from distributed.deploy.local import LocalCluster
	        cluster = LocalCluster()
	    return cluster

if __name__=='__main__':
	# start training
	classifier= Classifier(whether_client=OPT.client,window_size=OPT.window_size, workers= OPT.workers,
	 					verbose=OPT.verbose, grid_search=OPT.grid_search, model_path=OPT.model)
	if OPT.train:
		classifier.train()
	elif OPT.test:
		img='test2.png'
		src= cv2.imread(img)
		print(classifier.test(OPT.model, src))
