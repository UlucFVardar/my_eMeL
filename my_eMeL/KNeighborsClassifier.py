from collections import Counter
import pandas as pd
from DistanceFunctions import distance_calculate
import copy
import numpy as np


class KNeighborsClassifier():
    def __init__(self, k_number, distance_metric ):
        self.log = { 'success' : 0, 'error' : 0}
        self.k_number =  k_number
        self.distance_metric = distance_metric

    def fit(self, data,label,label_col_name):
        self.data = data
        self.label = label
        self.label_col_name = label_col_name
        
    #------
    def distance_calculate_all_df(self,df,point,distance_metric,label_col_name):
        def mini_function(row,p2,distance_metric,label_col_name):      
            p1 = np.array([float(x) for x in list(row)])
            return distance_calculate(p1,p2.astype(float),distance_metric)
        df[distance_metric] = df.apply(mini_function,args=(point,distance_metric,label_col_name,),axis=1)
        return df    
    def predict_test(self, test_data_df, test_label_df ):
        predictions = []
        test_data_numpy = test_data_df.values
        labels_of_point = test_label_df.values
        for one_point_in_test,label_of_point in zip(test_data_numpy,labels_of_point):
            #print one_point_in_test,label_of_point 

            new_ = self.distance_calculate_all_df(copy.deepcopy(self.data)  ,one_point_in_test,self.distance_metric,self.label_col_name)
            new_ = pd.concat( [new_, self.label ] ,axis=1)    
            new_ = new_.sort_values(by=[self.distance_metric])
            new_ = new_[:self.k_number]

            predictions.append(Counter(list(new_[self.label_col_name])).most_common(1)[0][0])
            if Counter(list(new_[self.label_col_name])).most_common(1)[0][0] == label_of_point[0]:
                self.log['success'] +=1
            else:
                self.log['error'] +=1
        return pd.DataFrame({'predictions':predictions})
    def predict(self, test_data_df ):
        predictions = []
        test_data_numpy = test_data_df.values
        for one_point_in_test in test_data_numpy:
            #print one_point_in_test,label_of_point 
            new_ = self.distance_calculate_all_df(copy.deepcopy(self.data)  ,one_point_in_test,self.distance_metric,self.label_col_name)
            new_ = pd.concat( [new_, self.label ] ,axis=1)    
            new_ = new_.sort_values(by=[self.distance_metric])
            new_ = new_[:self.k_number]
            predictions.append(Counter(list(new_[self.label_col_name])).most_common(1)[0][0])
        return pd.DataFrame({'predictions':predictions})

    def get_accuracy_values(self):
        return '%.2f'%(float(float(self.log['success'])/(self.log['error']+self.log['success']))*100),(str(self.log['error'])+'/'+str((self.log['error']+self.log['success'])))
