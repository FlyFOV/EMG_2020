import os
import numpy as np
import pandas as pd
from scipy.stats import skew
from tensorflow import keras
import pywt

SIZE = 4000

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

    #########################################################################
    # function name : wavelet decompostion
    # function description : extract wavelet information of the raw data
    #                       raw data
    # function parameters:  raw_data: a matrix with 8 column and n rows
    #
    #
    # author: Haotian Fan
    # time: 30 June 2020
    ########################################################################

    def wavelet_decompostion(self,raw_data):
        def wavlet(channel):

            coeffs = pywt.wavedec(channel,'db7',level = 4)
            cA4, cD4, cD3, cD2, cD1 = coeffs
            recD1 = pywt.waverec([cD1],'db7')
            recD2 = pywt.waverec([cD2],'db7')
            res = []
            a  = np.mean(np.absolute(cA4))
            b = np.mean(np.absolute(cD4))
            c = np.mean(np.absolute(cD3))
            d = np.mean(np.absolute(recD1))
            e = np.mean(np.absolute(recD2))
            res.append([a,b,c,d,e])
            res = np.array(res)
            return res
        res = []
        for col in raw_data.T:
            dec = wavlet(col)
            res.extend(dec[0])
        res = np.array(res)
        return res


    def mean_absoule_value(self,data):
        res = []
        for n in data.T:
            res.append(np.mean(np.absolute(n)))
        return res

    def root_mean_square(self,data):
        res = []
        for n in data.T:
            res.append(np.sqrt(np.sum(np.square(n)) / n.size))
        return res

    def waveform_length(self,data):
        res = []
        for n in data.T:
            res.append(np.sum(np.absolute(np.diff(n))))
        return res

    def willison_amplitude(self,data, threshold):
        res = []
        for n in data.T:
            res.append(np.sum(np.absolute(np.diff(n)) >= threshold))
        return res

    def skewness(self,data):
        res = []
        for n in data.T:
            res.append(np.absolute(skew(n)))
        return res

    #########################################################################
    # function name : feature extraction
    # function description :
    # function parameters:
    #
    #
    # author: Haotian Fan
    # time: 30 June 2020
    ########################################################################

    def get_feature(self,data):
        res = np.array([])
        count = 0
        start = 0
        for col in data:
            count+=1
            if count == SIZE:
                mat = data[start:start+SIZE]
                start = start+SIZE
                count=0
                feature = []
                feature.append(keras.utils.normalize(self.skewness(mat)).tolist())
                feature.append(keras.utils.normalize(self.mean_absoule_value(mat)).tolist())
                feature.append(keras.utils.normalize(self.willison_amplitude(mat,0.00002)).tolist())
                feature.append(keras.utils.normalize(self.waveform_length(mat)).tolist())
                feature.append(keras.utils.normalize(self.root_mean_square(mat)).tolist())
                feature = np.array(feature)
                feature = feature.flatten().tolist()

                wave = self.wavelet_decompostion(mat).tolist()

                feature = feature+wave
                # print(wave)
                # print(wave.shape)

                #feature.append(keras.utils.normalize(self.mean_absoule_value(wave)).tolist())
                feature = np.array(feature)

                res = np.append(res,feature)

                #
                # wave = np.array(wave)
                # feature.append(keras.utils.normalize(self.mean_absoule_value(wave)).tolist())
                #
                #
                # res = np.append(res,feature)
        return res

