import numpy as np
import cv2
import random
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D

categories = ["Dog","Cat"]
label = [0,1]
size = 50

epo = 3
ker = 3

training_data =list()
training_imgs = list()
training_labels = list()
count=0

for cate,lab in zip(categories,label):
    for i in range(0,12500):
        try:
            img = cv2.imread('/home/PetImages/'+cate+'/'+str(i)+'.jpg' , cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img,(size,size))
            training_data.append([new_img,lab])
        except:
            pass

random.shuffle(training_data)

for features,label in training_data:
    training_imgs.append(features)
    training_labels.append(label)

training_imgs = np.array(training_imgs).reshape(-1,size,size,1)
training_labels = np.array(training_labels).reshape(-1,1)

X=training_imgs/255.0
y=training_labels

model= Sequential()

model.add(Conv2D(32,(ker,ker),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(ker,ker)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X,y,batch_size=4,epochs=epo,validation_split=0.3,verbose=1)

model.save("CNN.model")

print(history.history['accuracy'][0] * 100)
f = open("accuracy.txt",'w')
f.write('%d' % int(history.history['accuracy'][0] * 100))
f.close()
