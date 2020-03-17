

import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(64,64,1), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(64, (3, 3),  padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3),  padding='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Conv2D(128, (3,3),  padding='same', activation='relu'))
model.add(Conv2D(128, (3,3),  padding='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Conv2D(256, (3,3),  padding='same', activation='relu'))
model.add(Conv2D(256, (3,3),  padding='same', activation='relu'))
model.add(Conv2D(256, (3,3),  padding='same', activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(21, activation='softmax'))

model.summary()

model.load_weights('lenet5_32.h5')

import cv2
import numpy as np

img = cv2.imread('h2.png',0)
img = cv2.resize(img, (32,32))
img = np.reshape(img, (1,32,32,1))
img.shape

p = model.predict(img)

np.argmax(p)

if letter <= 9:
	print(letter)
elif letter == 10:
	print('অ')
elif letter == 11:
	print('আ')
elif letter == 12:
	print('ই')
elif letter == 13:
	print('ঈ')
elif letter == 14:
	print('উ')
elif letter == 15:
	print('ঊ')
elif letter == 16:
	print('ঋ')
elif letter == 17:
	print('এ')
elif letter == 18:
	print('ঐ')
elif letter == 19:
	print('ও')
else:
	print('ঔ')

