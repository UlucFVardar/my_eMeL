import numpy as np
import pandas as pd
import random
import DistanceFunctions as DisF
import matplotlib.pyplot as plt
from collections import Counter

class KMeans_Clustring():
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
        self.savePlots = False
        self.iteration = 0
    def choose_rnd_point(self):
        self.centers = []
        indexes = []
        for i in range(0,self.n_clusters):
            index = random.randint(0, len(self.data))
            if index not in indexes:
                self.centers.append(self.data[index])
                indexes.append(index)
    def fit(self,data):
        def calculate_distance_and_create_temp_lables(centers):
            #-----
            temp_lables = []
            for data_point in zip(self.data):
                distances = []
                for i, center_point in enumerate(centers):
                    distance = DisF.distance_calculate(data_point,center_point,'Euclidean')
                    distances.append([ distance,i])
                temp_label = min(distances, key = lambda t: t[0])
                temp_lables.append(temp_label[1])
            #-----
            return temp_lables
        def calculate_objective_funciton(centers,temp_lables):
            #------ calculate objectife funciton 
            TotalSum = 0
            for i, center_point in enumerate(centers):
                centers_points = filter(lambda x: x[1] == i ,map(lambda p,l: [p,l] ,self.data,temp_lables ))
                distances = map(lambda x : DisF.distance_calculate(x[0],center_point,'Euclidean')**2,centers_points)
                sum_distance = sum(distances)
                #print sum_distance
                TotalSum += sum_distance
            #print TotalSum,'----'
            #------            
            return TotalSum
        def centers_optimize(centers,labels):
            newCenters = []
            for i, center_point in enumerate(centers):
                centers_points = map(lambda x: x[0],filter(lambda x: x[1] == i ,map(lambda p,l: [p,l] ,self.data,temp_lables )))
                #print len (centers_points)
                centers_points = np.array(centers_points)
                Xmean = np.mean(centers_points[:,0])
                Ymean = np.mean(centers_points[:,1])                
                newCenters.append([Xmean,Ymean])
            return newCenters
        def calculate_change_of_centers(center1,centers2):
            return map(lambda c1,c2 : DisF.distance_calculate(c1,c2,'Euclidean'), center1,centers2)
   
        #-------------------------- FIT START -------------------------------------
        self.data = data
        self.choose_rnd_point()
        self.initialCenters = self.centers
        #print self.centers 
        self.plot_clusters(self.data,[0 for i in range(0,len(self.data))],title = 'Initial Stage of Kmeans')
        self.Objective_Values = []
        #-----
        self.iteration = 0
        while True:
            temp_lables = calculate_distance_and_create_temp_lables(self.centers) 
            #print Counter(temp_lables)
            ObjectiveValue = calculate_objective_funciton(self.centers,temp_lables)
            self.Objective_Values.append(ObjectiveValue)
            #print ObjectiveValue
            newCenters = centers_optimize(self.centers,temp_lables)
            self.iteration+=1
            self.plot_clusters(self.data,temp_lables, newCenters,('%s. iteration of Kmeans K=%s'%(str(self.iteration),str(self.n_clusters))) )
            change_of_centers = calculate_change_of_centers(self.centers,newCenters)
            #print change_of_centers
            MAX_change = max(change_of_centers)
            self.centers = newCenters
            #print MAX_change
            if MAX_change < 0.0001:
                break
        #-----
        self.cluster_centers_ = self.centers
        self.lables_ = temp_lables
        self.plot_Objective_values()

    def plot_Objective_values(self):
        # Data
        fig = plt.figure()
        df=pd.DataFrame({'x': range(0,len(self.Objective_Values)), 'Objective function': [int(x) for x in self.Objective_Values] })
        fig =plt.gcf()
        fig.set_size_inches(11, 7)
        # multiple line plot
        plt.ylabel('Objective Function Value')
        plt.xlabel('Iteration Number')          
        plt.plot( 'x', 'Objective function', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
        if self.savePlots == False:
            plt.show()
        else:
            fig.savefig(self.savePlots+'Objective Function')

    def plot_clusters(self,data,labels,centers=None,title = None):
        import os

        data = np.array(data)
        fig = plt.figure()
        plt.ylabel('Y Range')
        plt.xlabel('X Range')        
        if title!=None:
            plt.title(title)
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[:,0],data[:,1],c=labels,s=10)
        if centers != None:
            for i,j in centers:
                ax.scatter(i,j,s=50,c='red',marker='+')
        fig =plt.gcf()
        fig.set_size_inches(11, 7)
        if self.savePlots == False:
            fig.show()
        else:
            os.system('mkdir %s'%(self.savePlots[:-1]))
            fig.savefig(self.savePlots+str(self.iteration))