import csv
import cv2
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Flatten, Dense,Lambda,Cropping2D,Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from random import shuffle
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			correction = 0.12 # this is a parameter to tune
			for batch_sample in batch_samples:
				center_image = cv2.imread(batch_sample[0])
				left_image = cv2.imread(batch_sample[1])
				right_image = cv2.imread(batch_sample[2])
				center_angle = float(batch_sample[3])
				left_angle = center_angle + correction
				right_angle = center_angle - correction
				images.extend([center_image,left_image,right_image])
				angles.extend([center_angle,left_angle,right_angle])
				images.extend([np.fliplr(center_image),np.fliplr(left_image),np.fliplr(right_image)])
				angles.extend([-center_angle,-left_angle,-right_angle])

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)
			
lines =[]
with open('data/driving_log.csv') as csv_file:
	reader = csv.reader(csv_file)
	for line in reader:
		lines.append(line)


train_samples, validation_samples = train_test_split(lines, test_size=0.2)

images = []
image_names = []
measurements = []

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

X_train = np.array(images)
y_train = np.array(measurements)

dropout_rate = 0.5

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(dropout_rate))
model.add(Dense(50))
model.add(Dropout(dropout_rate))
model.add(Dense(10))
model.add(Dropout(dropout_rate))
model.add(Dense(5))
model.add(Dropout(dropout_rate))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

history_object = model.fit_generator(train_generator, 
	samples_per_epoch=len(train_samples), 
	validation_data=validation_generator,
	nb_val_samples=len(validation_samples), 
	nb_epoch=3)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()