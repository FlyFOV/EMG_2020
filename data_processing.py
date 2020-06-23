import numpy as np
from scipy.stats import  skew
import os
import pandas as pd
import re
from pathlib import Path

############ global variables ###################
current_path = os.getcwd()
data_path = str(current_path+'/data/raw/')
print (data_path)
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
        self.labeled_raw_train_data = pd.DataFrame(columns = self.col )
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

    def collect_data(self,subject=0,index=[1,2], gesture = all_gesture):
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
                print(files)
                databuf = pd.read_csv(self.directory+str(files),header=None,names = self.col)
                print(databuf)
                databuf.insert(column='gesture',value = label_num,loc=8)
                print(databuf)
                self.labeled_raw_train_data = pd.concat([self.labeled_raw_train_data,databuf],ignore_index=True)

        return self.labeled_raw_train_data

Q = Get_Data()
print(Q.collect_data(subject=1,gesture = [0]))

