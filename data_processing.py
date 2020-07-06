import numpy as np
from scipy.stats import  skew
import os
import pandas as pd
import feature_extraction
import re
from pathlib import Path

############ global variables ###################
current_path = os.getcwd()
data_path = str(current_path+'/data/raw/')
figure_path = str(current_path+'/figure/')
############ global variables ###################

#################################################
#class name : Get_Data
#Parameter:
#Function list:
#description:
#aurthor: Haotian Fan
#Time:
#################################################
class Get_Data(object):
    all_gesture = [i for i in range(0,15)]

    def __init__(self ):
        self.col = ['ch0','ch1','ch2','ch3','ch4','ch5','ch6','ch7']
        self.raw_train_data = pd.DataFrame(columns = self.col )
        self.raw_train_label = np.array([],dtype = int)
        self.directory = data_path


    def label_to_number(self, label):
        return {
            'HC_': 0,
            'I_I': 1,
            'I_M': 2,
            'IMR': 3,
            'L_L': 4,
            'M_M': 5,
            'M_R': 6,
            'MRL': 7,
            'R_L': 8,
            'R_R': 9,
            'T_I': 10,
            'T_L': 11,
            'T_M': 12,
            'T_R': 13,
            'T_T': 14,
        }[label]

    #################################################
    # function name : collect data
    # Parameter:
    #           subject : choose the subject of experiment,default choose all the experiment subjects.
    #           index : choose the trail num of raw data. Mainly designed for choose train and test dataset.
    #           gesture : choose the hand gesture you want to training.
    #           time_frame : set the time frame , default 5s. The frequency of sampling data is 4000HZ,
    #                        which means 0.25ms one line . Range from 0s up to 20s.
    # description: read raw data from csv file and covert into DataFrame
    # aurthor: Haotian Fan
    # Time: 23 June 2020
    #################################################
    def collect_data(self,subject=0,index=[1], gesture = all_gesture,time_frame = 5):
        for files in os.listdir(self.directory):
            add_on = False
            label = files[3:6]
            sname = int(files[1])
            i_num = int(files[-5])
            label_num = self.label_to_number(label)

            if subject == 0:
                sname = 0
            if sname == subject and i_num in index and label_num in gesture:
                add_on = True
            if add_on:
                self.raw_train_label = np.append(self.raw_train_label,int(label_num))
                databuf = pd.read_csv(self.directory+str(files),header=None,names = self.col)

                #databuf.insert(column='gesture',value = label_num,loc=8)
                databuf = databuf[0:int((time_frame/20)*80000)]

                self.raw_train_data = pd.concat([self.raw_train_data,databuf],ignore_index=True)

        return self.raw_train_data.to_numpy(),self.raw_train_label



# Q = Get_Data()
# raw_data ,label_data= Q.collect_data(1,[1,2],[1],time_frame=1)
# print(raw_data)
# print (label_data)
# raw_data = raw_data.to_numpy()
# raw_data = raw_data.reshape(-1,4000,8)
# print (raw_data.shape)
#print (raw_data.shape)
# F = feature_extraction.feature_extraction()
# print (len(F.window_process(raw_data.to_numpy(),10,5)))
# print(F.window_process(raw_data.to_numpy(),10,5))

