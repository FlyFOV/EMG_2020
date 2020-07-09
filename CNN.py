import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,MaxPool2D
from sklearn.metrics import accuracy_score
import data_processing
import feature_extraction
import matplotlib.pyplot as plt
import os

###########global variables ############################################
TIMEFRAME = 5
SIZE = int(TIMEFRAME*80000/20)
WINDOW_SIZE = 4000
INTERVAL = 1000
SAMPLE_N_AF_WINDOW = int((SIZE - WINDOW_SIZE)/INTERVAL+1)
########################################################################

data_father = data_processing.Get_Data()
raw_data,raw_label = data_father.collect_data(0,[1,2],[0,1,2,3],time_frame=TIMEFRAME)
test_father = data_processing.Get_Data()
test_data,test_label = test_father.collect_data(0,[3],[0,1,2,3],time_frame=TIMEFRAME)


############# windowing the raw data ####################################
feature = feature_extraction.feature_extraction()
raw_data ,raw_label= feature.window_process(raw_data,WINDOW_SIZE,INTERVAL,SIZE,raw_label)
test_data,test_label = feature.window_process(test_data,WINDOW_SIZE,INTERVAL,SIZE,test_label)
#########################################################################

raw_data = raw_data.reshape(-1,WINDOW_SIZE,8,1)
test_data = test_data.reshape(-1,WINDOW_SIZE,8,1)
raw_label = data_father.label_data_conversion(raw_label,4)
print (raw_label)

regularizer = keras.regularizers.l2(l=0.00012)


########################## model setup #########################################
model = keras.models.Sequential()
model.add(Conv2D(5,(50,8),activation='tanh',input_shape=raw_data.shape[1:],padding='same',kernel_regularizer=regularizer,bias_regularizer=regularizer))
model.add(MaxPool2D(pool_size=(4,2)))
model.add(Dropout(0.5))
model.add(keras.layers.Conv2D(5, (20, 4), activation='tanh', padding='same',kernel_regularizer=regularizer,bias_regularizer=regularizer))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(keras.layers.Conv2D(5,(10,2), activation='tanh', padding='same',kernel_regularizer=regularizer,bias_regularizer=regularizer))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(200, activation='tanh',kernel_regularizer=regularizer,activity_regularizer=regularizer))
model.add(keras.layers.Dense(150, activation='tanh',kernel_regularizer=regularizer,activity_regularizer=regularizer))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4, activation=tf.nn.softmax))
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, verbose=0,
                                               mode='auto', baseline=None)


########################model setup #############################################


history = model.fit(raw_data, raw_label,
              epochs=1,
              batch_size=20,
              shuffle=False,
              callbacks=[early_stopping])

prediction = model.predict(test_data)
prediction = [np.argmax(k) for k in prediction]
print (accuracy_score(prediction,test_label))
print (history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

#plt.show()
plt.savefig(data_processing.figure_path+"cnn_acc.png" )