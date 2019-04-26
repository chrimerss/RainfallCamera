'''
This module is the GPU version of calculating rain intensity with Allamano algorithm

full information goes here: ../PReNet/RainProperty.py
'''

class Allamano(Base):

	def __init__(self, tensor, focal_len=20, ex_time=1/250, f_num=1.85, focus_dist=50,
				 sensor_h=2.8, threshold=0.08, verbose=False, graph=False):

		self.assert_tensor(tensor)
		self.tensor= tensor
		self.h, self.w= self.tensor.shape
		self.focal_len=focal_len
		self.ex_time= ex_time
		self.f_num= f_num
		self.focus_dist= focus_dist
		self.sensor_h= sensor_h
		self.threshold= threshold
		self.verbose= verbose
		self.graph= graph
		self.lengths= []
		self.diameters= []

	def streaklength(self, img, graph):
		'''
		Args:
		---------------
			img: cv2.UMat object;
			graph: bool, determines to show graph or not
		'''
		self.assert_UMat(img)
		contours, hierarchy= cv2.findContours(img, cv2.RETR_LIST, \
											cv2.CHAIN_APPROX_SIMPLE)
		gray_scale= self.cvt_BGR2GRAY(img)
		V= []
		for contour in contours:
			M= cv2.moments(contour)
			x,y,w,h = cv2.boundingRect(contour)
			self.assert_UMat(x,y,w,h)
			font=cv2.FONT_HERSHEY_SIMPLEX
			if w != 0 and h !=0:
				length= np.sqrt(w**2+h**2)
				area = cv2.contourArea(contour)
				self.assert_UMat(area, length)
				length, area= self.umat2tensor(length), self.umat2tensor(area)
				diameter = self._Pixel2mm(area/length)
				if diameter<self.streak_diameter:
					self.diameters.append(diameter)
					self.length.append(length)
					dist= self.cal_dist(length, diameter)
					v= self._cal_drop_v(diameter, dist)
					V.append(v)
					cv2.rectangle(gray_scale, (x,y), (x+w, y+h), (0,0,255),1)
					cv2.putText(colored_img,f'{int(length)}', (x,y),font,0.2,(0,0,255),1,cv2.LINE_AA)
					
		if graph:
			cv2.imshow('rainstreak', gray_scale)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		self.diameters= torch.Tensor(self.diameters).cuda()
		self.lengths= torch.Tensor(self.lengths).cuda()

		return torch.Tensor(V)

	def cal_dist(self,length,diameter, silent=True):
		length_mm= self._pixel2mm(length)
		diameter_mm= self._pixel2mm(diameter)
		if diameter_mm<self.A:
			alpha = (length-diameter)/self.ex_time/1000*self.sensor_h/self.focal_len/self.h
			beta = 21.62*diameter*self.sensor_h/self.focal_len/self.h
			gamma = 21.62*self.A/1000/self.focus_dist*(1-2*self.threshold/self.del_l)
			self.assert_tensor(alpha, beta, gamma)
			dist = self._dist_fun(self.focus_dist, alpha, beta, gamma)

		return dist

	def _dist_fun(self, z0, alpha, beta, gamma):
		func= alpha**2*z**2-beta*z+gamma*abs(z-z0)
		d1= ((beta-gamma)+((beta-gamma)**2+4*alpha**2*z0*gamma)**.5)/2/ahpha**2
		d2= ((beta-gamma)-((beta-gamma)**2+4*alpha**2*z0*gamma)**.5)/2/ahpha**2
		if d1>0 and d2>0:
			return [d1 if torch.randint(0,2,1)==0 else d2]
		elif (d1>0 and d2<0):
			return d1
		elif (d1<0 and d2>0):
			return d2
		else:
			raise ValueError('no root find!')


	def _pixel2mm(self, pix):
		d2= 20*1000
		d1= self.focal_len*d2/(d2-self.focal_len)
		h_mm= self.focal_len/d1*3000
		pixel2mm= h_mm/self.h

		return pixel2mm

	def _control_volumne(self):

		return 52/81*(self.focus_dist*self._Pixel2mm(self.w)/self._Pixel2mm(self.h)*4*(self._Pixel2mm(self.h)/1000)**2)

	def _cal_drop_v(self, diameter, length):
		try:
			d_p = self.sensor_h/self.focal_len/self._pixel2mm(self.h)*np.array(dist)*1000
			v = np.sqrt(21.62*diameter*d_p)

			return v

		except ValueError:
			print('distance cannot be calculated ', dist,f'\nsensor_h: {self.sensor_h}, focal length: {self.focal_len} h: {self.h}')

	def rainrate(self):

		V= self.streaklength(self.tensor)