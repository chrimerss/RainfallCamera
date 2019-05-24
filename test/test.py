import pytest
import configparser
import sys
sys.path.append('../PReNet')
from rainproperty import RainProperty

#======load configuration======
config= configparser.ConfigParser()
config.read('../camera.ini')

def test_camera_parameter():
	# build instance
	global config
	NC450= config['NC450']
	rain= RainProperty(mat=)

	assert NC450['focal_len']==
