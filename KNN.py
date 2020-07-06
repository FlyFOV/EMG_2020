from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import data_processing

data_father = data_processing.Get_Data()
raw_data = data_father.collect_data(0,[1,2],[0,1,2],time_frame=5)
test_data = data_father.collect_data(0,[3],[0,1,2],time_frame=5)

train_label = raw_data['gesture'].to_numpy()
train_data = raw_data.to_numpy()[:,:8]
print (train_data.shape)
test_label = test_data['gesture'].to_numpy()
test_data = test_data.to_numpy()[:,:8]


neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(train_data,train_label)

prediction = neigh.predict(test_data)
print(test_label.shape)
print (accuracy_score(prediction,test_label))