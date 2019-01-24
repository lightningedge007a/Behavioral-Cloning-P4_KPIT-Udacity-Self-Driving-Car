import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Dropout
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D,Convolution2D
from keras.callbacks import ModelCheckpoint
df=pd.read_csv('Data/MyDrivingDataTrack/driving_log.csv',names=['Center Image','Left Image','Right Image','Steering Angle','Throttle','Break','Speed'])

print(df.shape)

model=Sequential()
model.add(Lambda(lambda x:(x/127.5)-1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0)))) # 3x80x320
#Conv2D(36, (5, 5), strides=(2, 2))
model.add(Conv2D(24,(5,5),strides=(2,2),activation='elu'))
model.add(Conv2D(36,(5,5),subsample=(2,2),activation='elu'))
model.add(Conv2D(48,(5,5),subsample=(2,2),activation='elu'))
model.add(Conv2D(64,(3,3),activation='elu'))
model.add(Conv2D(64,(3,3),activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
#model.add(Dense(1164,activation='elu'))
model.add(Dense(100,activation='elu'))
model.add(Dense(50,activation='elu'))
model.add(Dense(10,activation='elu'))
model.add(Dense(1))

import os
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(df, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #Shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            #
            images=[]
            measurements=[]
            for i in range(len(batch_samples)):

              line=batch_samples.iloc[i]['Center Image']
              linel=batch_samples.iloc[i]['Left Image']
              liner=batch_samples.iloc[i]['Right Image']

              name=line.split('\\')[-1]
              namel=linel.split('\\')[-1]
              namer=liner.split('\\')[-1]

              path='Data/MyDrivingDataTrack/IMG/'+name
              pathl='Data/MyDrivingDataTrack/IMG/'+namel
              pathr='Data/MyDrivingDataTrack/IMG/'+namer

              img=cv2.imread(path)
              img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              imgflip=np.fliplr(img)

              imgl=cv2.imread(pathl)
              imgl=cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)
              imgflipl=np.fliplr(imgl)

              imgr=cv2.imread(pathr)
              imgr=cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
              imgflipr=np.fliplr(imgr)


              images.extend([img,imgl,imgr,imgflip,imgflipl,imgflipr])

              angle=float(df.iloc[i]['Steering Angle'])
              angleflip=(-1)*angle

              corr=0.2

              anglel=angle+corr
              angleflipl=(-1)*anglel

              angler=angle-corr
              angleflipr=(-1)*angler

              measurements.extend([angle,anglel,angler,angleflip,angleflipl,angleflipr])

            #
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format
model.compile(loss='mse',optimizer='adam')

checkpoint = ModelCheckpoint('model2-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

history_object=model.fit_generator(train_generator, steps_per_epoch= len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose = 1,callbacks=[checkpoint])

model.save('model2.h5')

