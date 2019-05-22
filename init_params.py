'''
This file initialize camera parameters and system settings.

All the parameters should be written in camera.ini
'''
import sys
sys.path.append('./PReNet')
import configparser
import numpy as np


config = configparser.ConfigParser()

#========= Camera 1 (cloud) Parameters============
config['cloud_camera']= {'focal_len': 20,
						'ex_time': 1/250.,
						'focus_dist': 50,
						'f_num': 1.85,
						'sensor_h': 2.8,
						'threshold': 0.08,
						'del_l': 50,
						'streak_diameter': 10,
						'resolution': 'Full-HD',
						'fps': 1}

#========= Camera 2 (NC450) Parameters============
#source: https://www.tp-link.com/sg/home-networking/cloud-camera/nc450/#specifications

config['NC450']= {"focal_len":4,
				'ex_time':1/60,
				'sensor_h':6.35,
				'f_num':3.6,
				'focus_dist': 20,
				'threshold':0.08,
				'streak_diameter':10,
				'del_l':50,
				'fps':15,
				'resolution':str((1280,720))}

#========= classification configuration =========
config['classification']= {'model_path': 'D:\\CCTV\\rainfallcamera\\classification\\svm_model-4-grid_searched-200x200.joblib',
							'patches': str((20,20)),
							'window_size': str((10,10)),
							'information': str(['contract', 'minimum brightness', 
											'sharpness', 'mean hue', 'mean saturation']),
							}

#============== RNN configuration =================
config['PReNet']= {'model_path': 'D:\\CCTV\\rainfallcamera\\PReNet\\logs\\real\\PReNet1.pth',
						'recurrent_iter': 4,
						'logdir': 'D:\\CCTV\\rainfallcamera\\PReNet\\logs\\real',
						'use_GPU': True,
						'folder':str(20180401),
						'gpu_id':str(0)}


# Write configuration file
with open('camera.ini', 'w') as f:
	config.write(f)