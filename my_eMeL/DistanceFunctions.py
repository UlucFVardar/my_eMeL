import numpy as np
import math

def get_AllDistanceFuncitonsName():
	return ['Cosine',
			'Euclidean',
			'Manhattan',
			'my_Cosine']



def distance_calculate(p1,p2,distance_metric):
	def cosDist(p1, p2):

		Orjinn = np.array([0., 0.])
		e1 = p1 - Orjinn 
		#---
		tmp = 180 / np.pi 
		angles = []
		#--
		e2 = p2 - Orjinn
		num = np.dot(e1, e2)
		denom = np.linalg.norm(e1) * np.linalg.norm(e2)
		return np.arccos(num/denom) * tmp
	def my_cosineDistance(p1,p2):
	    
	    def calclualte_len(p):
	        return math.sqrt((p[0]**2)+(p[1]**2))
	    def my_arccos(cos):
	        return math.acos(cos)
	    cos = float(p1[0]*p2[0] + p1[1]*p2[1])/float(calclualte_len(p1)*calclualte_len(p2))
	    return my_arccos(cos)*(180/math.pi)


	if distance_metric == 'Euclidean':
		return np.linalg.norm((p1-p2))
	if distance_metric == 'Manhattan':
		return np.linalg.norm((p1-p2),ord=1)
	if distance_metric == 'my_Cosine':
		try:
			return my_cosineDistance(p1,p2)
		except Exception as e:
			return 0.0 # same points
	if distance_metric == 'Cosine':
		return cosDist(p1,p2)






