import sys
sys.path.append("..")
import flask
from flask import Flask, request, render_template,url_for, flash
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import cv2
import os
import scipy.misc
import torch
from PReNet.generator import Generator_lstm
from PReNet.execute_cpu import RRCal
# config:
UPLOAD_FOLDER= os.path.join('static','uploaded')
ALLOWED_EXTENSIONS= set(['png','jpg','jpeg'])

app= Flask(__name__)
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
	if request.method=='POST':
		file= request.files['image']
		if not file or not allowed_file(file.filename):
			flash('No file uploaded!')
			return render_template('index.html', label='need to upload file first!')
		elif file and allowed_file(file.filename):
			img= scipy.misc.imread(file)
			filename = secure_filename(file.filename)
			scipy.misc.imsave(os.path.join(app.config['UPLOAD_FOLDER'], filename), img)
			src=cv2.resize(img, (200,200)).copy()
			info= moving_window(src)
			model= load('../classification/svm_model-4-grid_searched-200x200.joblib')
			label=model.predict(info)[0]
			if label=='normal':
				intensity= cal_intensity(img[600:1000,300:600,:])
				intensity= round(intensity,2)
				# intensity='unknown'
			elif label=='night':
				intensity='unknown'
			elif label=='heavy':
				intensity= 'unknown'
			elif label=='no rain':
				intensity=0

			return render_template('index.html', label=label, value=intensity,
									 user_image= os.path.join(app.config['UPLOAD_FOLDER'], file.filename))



def moving_window(src, window_size=(10,10)):
	h,w, _= src.shape
	n_h, n_w= h//window_size[0], w//window_size[1]
	values=np.zeros((n_h*n_w, 5), dtype=np.float32)
	ind=0
	for i in range(n_h):
		for j in range(n_w):
			sub_img= src[i*window_size[0]: (i+1)*window_size[0], j*window_size[1]: (j+1)*window_size[1],:]
			ii,jj,mm= sub_img.shape
			err=0
			for m in range(mm):
				err+= (sub_img[:,:,m]-sub_img.mean(axis=2))**2
			values[ind,0]= err.sum()/ii/jj/mm
			values[ind,1]= sub_img.min()
			values[ind,2]= cv2.Laplacian(sub_img,cv2.CV_64F).var()
			values[ind,3]= cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)[:,:,0].mean()
			values[ind,4]= cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)[:,:,1].mean()
			ind+=1

	return values.reshape(1,-1)

def RNN_model(model_path='../PReNet/logs/real/PReNet1.pth',recur_iter=4):
	model= Generator_lstm(recur_iter, use_GPU=False)
	model.load_state_dict(torch.load(model_path, map_location='cpu'))
	model.eval()

	return model

def cal_intensity(img):
	model= RNN_model()

	return RRCal().img_based_im(img, model,use_GPU=False)

if __name__ == '__main__':
    app.secret_key = 'super secret key'

    app.run(host='0.0.0.0', port=8000, debug=True)
	