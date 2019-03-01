#!/usr/bin/env python
# coding: utf-8

# In[1]:


import my_eMeL.data_loader as data_loader
import my_eMeL.my_eMeL as my_eMeL


# In[2]:


# Loading data
data   = data_loader.load_known_txt_V2(  file_path           = './data1.txt'
                                              , delimiter          = ','
                                              , data_column_asList = [0,1]   )
# Pre-processing data
data   = map(lambda x : [float(x[0]),float(x[1])] ,data)

#create Model
kmeans = my_eMeL.KMeans(n_clusters=3)
kmeans.savePlots = './Kmeans-data1-k3/'
kmeans.fit(data)
#--------------------------------------
#create Model
kmeans = my_eMeL.KMeans(n_clusters=7)
kmeans.savePlots = './Kmeans-data1-k7/'
kmeans.fit(data)


# In[3]:


# Loading data
data   = data_loader.load_known_txt_V2(  file_path           = './data2.txt'
                                              , delimiter          = ','
                                              , data_column_asList = [0,1]   )
# Pre-processing data
data   = map(lambda x : [float(x[0]),float(x[1])] ,data)

#create Model
kmeans = my_eMeL.KMeans(n_clusters=2)
kmeans.savePlots = './Kmeans-data2-k2/'
kmeans.fit(data)
#--------------------------------------
#create Model
kmeans = my_eMeL.KMeans(n_clusters=5)
kmeans.savePlots = './Kmeans-data2-k5/'
kmeans.fit(data)


# In[4]:


# Loading data
data   = data_loader.load_known_txt_V2(  file_path           = './data3.txt'
                                              , delimiter          = ','
                                              , data_column_asList = [0,1]   )
# Pre-processing data
data   = map(lambda x : [float(x[0]),float(x[1])] ,data)

#create Model
kmeans = my_eMeL.KMeans(n_clusters=3)
kmeans.savePlots = './Kmeans-data3-k3/'
kmeans.fit(data)
#--------------------------------------
#create Model
kmeans = my_eMeL.KMeans(n_clusters=8)
kmeans.savePlots = './Kmeans-data3-k8/'
kmeans.fit(data)


# In[ ]:




