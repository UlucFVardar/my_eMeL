import pandas as pd
import numpy as np
from collections import Counter
import copy



from KNeighborsClassifier import KNeighborsClassifier as knc
from AccuracyTable import AccuracyTable
from KmeansClustering import KMeans_Clustring as KMeans_Clustring

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def KNeighborsClassifier( k_number, distance_metric ):
    return knc( k_number, distance_metric )      


def create_AccuracyTable( index, columns ):
    return AccuracyTable(index, columns)


def split_Train_and_Test( data, label, label_col_name, uniq_lables, first_n_number_train ):
    colnames = list(data.columns)
    data = pd.concat( [data, label ] ,axis=1)
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for uniq_label in uniq_lables:
        d = data[data[label_col_name] == uniq_label]
        train_df = pd.concat( [ d.iloc[:first_n_number_train,:], train_df]  ,axis=0)
        test_df =  pd.concat( [ d.iloc[first_n_number_train:,:] , test_df]  ,axis=0)
    train_df
    return train_df[colnames], train_df[[label_col_name]],\
            test_df[colnames],  test_df[[label_col_name]]

def draw_decisionBoundries (train_data_df, train_label_df, label_col_name, k, distance_metric_for_clf , h = 0.02):
    X = train_data_df.values.astype(float)
    y = (train_label_df[label_col_name].values).astype(int).tolist()

    # Create color maps
    cmap_light = ListedColormap(['#EE82EE', '#00BFFF', '#98FB98'])
    cmap_bold = ListedColormap(['#FF00FF', '#00008B', '#00FF00'])
    
    clf = knc( k_number        = k  ,
               distance_metric = distance_metric_for_clf   )
    clf.fit(  data  = train_data_df,
              label = train_label_df,
              label_col_name = label_col_name)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(  pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])   ).values.astype(int)

    
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("k = %i, Distance Metric = '%s'" % (k, distance_metric_for_clf))
    plt.ylabel('Iris_Feature_4')
    plt.xlabel('Iris_Feature_1')

    plt.show()    


def KMeans( n_clusters ):
    return KMeans_Clustring(n_clusters)    