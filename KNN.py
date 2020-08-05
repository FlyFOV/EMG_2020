from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import feature_extraction
import data_processing



TIMEFRAME = 5
SIZE = int(TIMEFRAME*80000/20)
WINDOW_SIZE = 4000
INTERVAL = 1000
SAMPLE_N_AF_WINDOW = int((SIZE - WINDOW_SIZE)/INTERVAL+1)


data_father = data_processing.Get_Data()
raw_data,raw_label = data_father.collect_data(0,[1,2],time_frame=5)
test_data,test_label = data_father.collect_data(0,[3],time_frame=5)

feature = feature_extraction.feature_extraction()
raw_data ,raw_label= feature.window_process(raw_data,WINDOW_SIZE,INTERVAL,SIZE,raw_label)
test_data,test_label = feature.window_process(test_data,WINDOW_SIZE,INTERVAL,SIZE,test_label)
print (raw_data.shape)
print (raw_label.shape)
fea = feature.get_feature(raw_data)
print(fea.shape)
fea = fea.reshape(-1,80)
print (fea.shape)

testfea = feature.get_feature(test_data)
testfea = testfea.reshape(-1,80)
# print (raw_data.shape)
# print (raw_label.shape)
# test_data = test_data.reshape(-1,WINDOW_SIZE,8)
# raw_data = raw_data.reshape(-1,WINDOW_SIZE,8)
# print (raw_data.shape)
# print (raw_label.shape)
train_data = [n.flatten() for n in raw_data]
test_data = [n.flatten() for n in test_data]
print (raw_data.shape)
print (raw_label.shape)
neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(fea,raw_label)

prediction = neigh.predict(testfea)
print(test_label.shape)
print (accuracy_score(prediction,test_label))
disp = plot_confusion_matrix(neigh,testfea,test_label,cmap=plt.cm.Blues)
plt.show()