This project is the source code of EMG signal classification based on supervised and unsupervised learning methods of Haotian Fan in Kings College London.
To run the source code, you need to prepare your environment.
Python is necessary to run the code, version 3.8 is recommended.
The following libraries are needed:

numpy
pywavelet
pandas
scipy
tensorflow
pickle
sklearn
matplotlib


After the installation of the libraries, you are able to run the source code.

For example,to run "knn" method, just open terminal and type the following command:
python KNN.py
All other methods have similary process.

It's necessary to methion that to run Kmeans method, there are two files. One is "kmeans.py",this file will collect the origin
EMG data and feed it to Kmeans classifier and save the model as a file "kmeans.sav" in your workspace.Another file is "kmeans_process",
this file calculate the accuracy of K-means clustering method. You have to run "kmeans.py" befor you run "kmeans_process".

Before you start running the code, check the function "data_collection" in "data_processing.py". This function selects the EMG raw data.
You can choose different gestures and subjects by using this function.

If you want to change the features used in this article,just heading into "get_feature" function in "feature_extraction.py". By comment and 
uncomment the feature selection code, you can get whatever feature you want. But after this procedure, remeber that the size of feature matrix 
has changed. So you need to reshape the feature matrix before you put it into the classifier.

As for the robustness test,if you want to test the classifier anti-noise ability, just uncommon "add_noise" function in "collect_data" function.
You can choose the level of noise. The noise is generated randomly, you can implement it serval time and choose the average value as the final result. 

