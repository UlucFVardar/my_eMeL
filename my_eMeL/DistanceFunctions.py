import numpy as np


def get_AllDistanceFuncitonsName():
	return ['Cosine',
			'Euclidean',
			'Manhattan']



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



	if distance_metric == 'Euclidean':
		return np.linalg.norm((p1-p2))
	if distance_metric == 'Manhattan':
		return np.linalg.norm((p1-p2),ord=1)
	if distance_metric == 'Cosine':
		return cosDist(p1,p2)





