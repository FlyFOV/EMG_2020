import os
import numpy as np
import pandas as pd


class feature_extraction(object):
    def __init__(self):
        self.mode = 0
#########################################################################
#function name : window_process
#function description : Using  a sliding window stragegy to process
#                       raw data
#function parameters:  raw_data: numpy data
#                      size: size of the window
#                      interval: gap between windows
#author: Haotian Fan
#time: 30 June 2020
########################################################################

    def window_process(self,raw_data,size,interval,raw_size,raw_label):
        ans = np.array([[]])
        window_label = []
        window_num = int((raw_size - size)/interval+1)
        for n in raw_label:
            for _ in range(window_num):
                window_label.append(int(n))
        window_label = np.array(window_label)
        for count in range(int(len(raw_data)/raw_size)):
            window_list = np.array([[]])
            curloc = 0
            cur_data = raw_data[count*(raw_size):(count+1)*(raw_size)]
            while curloc <= raw_size-size:
                if curloc == 0:
                    window_list = cur_data[curloc:curloc + size][:]
                else:
                    window_list = np.concatenate((window_list,cur_data[curloc:curloc + size][:]))
                curloc += interval
            ans = window_list if count ==0 else np.concatenate((ans,window_list))
        return ans,window_label




