

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout,  Flatten, MaxPooling2D
from keras.models import Sequential 
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.utils import to_categorical
import pandas as pd

x = np.load('all_x.npy')
print(x.shape, 'x')

y = pd.read_csv('y.csv').iloc[:,:].values

y1 = np.array([20])
y = np.vstack((y,y1))

y.shape

y = to_categorical(y)

print(y.shape)

x = np.expand_dims(x, axis=3)
x.shape


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

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
h = model.fit(X_train, y_train, validation_data = (X_test,y_test), batch_size=100, epochs=40)

history = h

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save_weights('lenet5_32.h5')

from google.colab import files
files.download('lenet5_32.h5')

